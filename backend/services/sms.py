"""
SMS alert engine.

* ``SmsProvider`` - the delivery interface.
* ``MockSmsProvider`` - the offline / test default; records to an in-memory
  outbox, returns ``MOCKED``. ``fail_next`` makes the next send fail (for
  retry tests).
* ``TwilioSmsProvider`` - the real implementation, driven by
  ``TWILIO_ACCOUNT_SID`` / ``TWILIO_AUTH_TOKEN`` / ``TWILIO_FROM_NUMBER``.
  Delivery failures are RETURNED as a FAILED ``SmsResult``, not raised, so the
  caller records the attempt and can retry.
* ``dispatch_alert`` - fingerprints the alert (location grid + time bucket +
  geometry signature), suppresses duplicates inside the cooldown, persists a
  full audit trail (``attempts`` / ``last_error`` / ``error_log``), and sends.
* ``retry_alert`` - re-attempt a FAILED alert.
* ``send_test_sms`` - deliver a test message to a user's own registered number
  (bypasses dedup).
"""
from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.alert import Alert, AlertChannel, AlertStatus
from backend.models.user import User

log = get_logger("backend.services.sms")


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
@dataclass
class SmsResult:
    ok: bool
    provider: str
    status: str                     # SENT | FAILED | MOCKED
    detail: dict = field(default_factory=dict)
    error: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SmsProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def send(self, *, to: str, body: str) -> SmsResult: ...


class MockSmsProvider(SmsProvider):
    """Records messages in memory; never touches the network."""

    name = "mock"

    def __init__(self) -> None:
        self.outbox: list[dict] = []
        self.fail_next: bool = False

    def send(self, *, to: str, body: str) -> SmsResult:
        if self.fail_next:
            self.fail_next = False
            log.warning("MOCK SMS forced failure -> %s", to)
            return SmsResult(ok=False, provider=self.name, status="FAILED",
                             error="forced mock failure", detail={"to": to})
        record = {"to": to, "body": body, "at": datetime.now(timezone.utc).isoformat()}
        self.outbox.append(record)
        log.info("MOCK SMS -> %s: %s", to, body[:80])
        return SmsResult(ok=True, provider=self.name, status="MOCKED", detail=record)


class TwilioSmsProvider(SmsProvider):
    """Real Twilio REST send. Constructed only when credentials are configured."""

    name = "twilio"

    def __init__(self) -> None:
        missing = [
            k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
            if not getattr(settings, k)
        ]
        if missing:
            raise RuntimeError(f"SMS_PROVIDER=twilio but {', '.join(missing)} not set")
        self._sid = settings.TWILIO_ACCOUNT_SID
        self._token = settings.TWILIO_AUTH_TOKEN
        self._from = settings.TWILIO_FROM_NUMBER

    def send(self, *, to: str, body: str) -> SmsResult:
        try:
            from twilio.rest import Client
        except ImportError:
            return SmsResult(ok=False, provider=self.name, status="FAILED",
                             error="the 'twilio' package is not installed")
        try:
            client = Client(self._sid, self._token)
            msg = client.messages.create(to=to, from_=self._from, body=body)
            return SmsResult(
                ok=True, provider=self.name, status="SENT",
                detail={"sid": msg.sid, "status": getattr(msg, "status", None)},
            )
        except Exception as exc:  # noqa: BLE001 - a delivery failure is data, not a crash
            log.warning("Twilio send failed -> %s: %s", to, exc)
            return SmsResult(ok=False, provider=self.name, status="FAILED",
                             error=f"{type(exc).__name__}: {exc}")


_provider: SmsProvider | None = None


def get_sms_provider() -> SmsProvider:
    """Process-wide provider chosen by ``settings.SMS_PROVIDER`` (mock | twilio)."""
    global _provider
    if _provider is None:
        _provider = (TwilioSmsProvider() if settings.SMS_PROVIDER.lower() == "twilio"
                     else MockSmsProvider())
        log.info("SMS provider = %s", _provider.name)
    return _provider


def reset_sms_provider() -> None:
    """Test hook - force ``get_sms_provider`` to re-read the config."""
    global _provider
    _provider = None


# --------------------------------------------------------------------------- #
# Fingerprinting + dedup
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _geometry_signature(geometry: dict | None) -> str:
    """A coarse, order-independent signature of a GeoJSON geometry."""
    if not geometry:
        return "-"
    try:
        from shapely.geometry import shape

        g = shape(geometry)
        w, s, e, n = (round(v, 2) for v in g.bounds)
        return f"{w},{s},{e},{n},{round(g.area, 3)}"
    except Exception:  # noqa: BLE001
        return hashlib.sha1(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:12]


def alert_fingerprint(
    *,
    lat: float | None,
    lon: float | None,
    occurred_at: datetime | None = None,
    geometry: dict | None = None,
    kind: str = "anomaly",
    grid_deg: float | None = None,
    time_bucket_minutes: int | None = None,
) -> tuple[str, dict]:
    """Deterministic fingerprint for an alert-worthy event."""
    grid = float(grid_deg or settings.ALERT_DEDUP_GRID_DEG)
    bucket_min = int(time_bucket_minutes or settings.ALERT_DEDUP_TIME_BUCKET_MINUTES)

    glat = round(round(float(lat) / grid) * grid, 4) if lat is not None else "na"
    glon = round(round(float(lon) / grid) * grid, 4) if lon is not None else "na"
    when = (occurred_at or _now())
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    bucket_idx = int(when.timestamp() // (bucket_min * 60))
    time_bucket = f"{bucket_min}m:{bucket_idx}"
    geo_sig = _geometry_signature(geometry)

    raw = f"{kind}|{glat}|{glon}|{time_bucket}|{geo_sig}"
    fp = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return fp, {
        "kind": kind, "grid_lat": glat, "grid_lon": glon,
        "time_bucket": time_bucket, "geometry_signature": geo_sig, "raw": raw,
    }


def find_duplicate(
    db: Session,
    fingerprint: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    cooldown_minutes: int | None = None,
) -> Alert | None:
    """A recent, non-suppressed alert this event should collapse into."""
    cooldown = int(cooldown_minutes if cooldown_minutes is not None
                   else settings.ALERT_DEDUP_COOLDOWN_MINUTES)
    since = _now() - timedelta(minutes=cooldown)
    live = (AlertStatus.SENT, AlertStatus.MOCKED, AlertStatus.PENDING)

    exact = db.execute(
        select(Alert)
        .where(Alert.fingerprint == fingerprint, Alert.created_at >= since,
               Alert.status.in_(live), Alert.dedup_of_id.is_(None))
        .order_by(Alert.id.desc())
    ).scalars().first()
    if exact is not None:
        return exact

    # geometry-similarity fallback: same coarse location within the cooldown
    if lat is not None and lon is not None:
        grid = float(settings.ALERT_DEDUP_GRID_DEG)
        near = db.execute(
            select(Alert)
            .where(Alert.created_at >= since, Alert.status.in_(live),
                   Alert.dedup_of_id.is_(None), Alert.lat.isnot(None))
            .order_by(Alert.id.desc())
        ).scalars().all()
        for a in near:
            if abs(a.lat - lat) <= grid and abs(a.lon - lon) <= grid:
                return a
    return None


# --------------------------------------------------------------------------- #
# Dispatch / retry
# --------------------------------------------------------------------------- #
def _append_log(alert: Alert, entry: dict) -> None:
    log_list = list(alert.error_log or [])
    log_list.append(entry)
    alert.error_log = log_list


def _send_now(db: Session, alert: Alert, provider: SmsProvider) -> Alert:
    alert.attempts = (alert.attempts or 0) + 1
    alert.provider = provider.name
    res = provider.send(to=alert.recipient, body=alert.message)
    entry = {
        "at": res.at.isoformat(), "attempt": alert.attempts, "provider": provider.name,
        "status": res.status, "error": res.error, "provider_response": res.detail,
    }
    _append_log(alert, entry)
    alert.provider_response = res.detail
    if res.ok:
        alert.status = AlertStatus.SENT if res.status == "SENT" else AlertStatus.MOCKED
        alert.sent_at = res.at
        alert.last_error = None
    else:
        alert.status = AlertStatus.FAILED
        alert.last_error = res.error
    db.commit()
    db.refresh(alert)
    return alert


def dispatch_alert(
    db: Session,
    *,
    message: str,
    recipient: str,
    lat: float | None = None,
    lon: float | None = None,
    occurred_at: datetime | None = None,
    geometry: dict | None = None,
    jurisdiction_codes: list[str] | None = None,
    investigation_id: int | None = None,
    anomaly_id: int | None = None,
    recipient_user_id: int | None = None,
    triggered_by_confidence: float | None = None,
    channel: AlertChannel = AlertChannel.SMS,
    kind: str = "anomaly",
    cooldown_minutes: int | None = None,
    provider: SmsProvider | None = None,
    force: bool = False,
    record_dedups: bool = True,
) -> Alert:
    """Fingerprint, dedup (unless ``force``), persist the audit row, and send."""
    fp, fp_meta = alert_fingerprint(
        lat=lat, lon=lon, occurred_at=occurred_at, geometry=geometry, kind=kind
    )

    if not force:
        dup = find_duplicate(db, fp, lat=lat, lon=lon, cooldown_minutes=cooldown_minutes)
        if dup is not None:
            log.info("alert suppressed as a duplicate of #%s (fp=%s)", dup.id, fp)
            if not record_dedups:
                return dup
            suppressed = Alert(
                investigation_id=investigation_id, anomaly_id=anomaly_id,
                recipient_user_id=recipient_user_id, channel=channel,
                status=AlertStatus.SUPPRESSED, recipient=recipient, message=message,
                triggered_by_confidence=triggered_by_confidence,
                lat=lat, lon=lon, geometry=geometry, jurisdiction_codes=jurisdiction_codes,
                fingerprint=fp, time_bucket=fp_meta["time_bucket"],
                dedup_of_id=dup.id, attempts=0, error_log=[],
                provider_response={"deduplicated": True, "original_alert_id": dup.id,
                                   "fingerprint_meta": fp_meta},
            )
            db.add(suppressed)
            db.commit()
            db.refresh(suppressed)
            return suppressed

    alert = Alert(
        investigation_id=investigation_id, anomaly_id=anomaly_id,
        recipient_user_id=recipient_user_id, channel=channel,
        status=AlertStatus.PENDING, recipient=recipient, message=message,
        triggered_by_confidence=triggered_by_confidence,
        lat=lat, lon=lon, geometry=geometry, jurisdiction_codes=jurisdiction_codes,
        fingerprint=fp, time_bucket=fp_meta["time_bucket"], attempts=0, error_log=[],
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return _send_now(db, alert, provider or get_sms_provider())


def retry_alert(db: Session, alert_id: int, *, provider: SmsProvider | None = None,
                max_attempts: int | None = None) -> Alert:
    """Re-attempt delivery of a FAILED alert."""
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise ValueError(f"alert {alert_id} not found")
    cap = int(max_attempts if max_attempts is not None else settings.ALERT_MAX_SEND_ATTEMPTS)
    if alert.status not in (AlertStatus.FAILED, AlertStatus.PENDING):
        return alert
    if (alert.attempts or 0) >= cap:
        alert.last_error = f"retry cap ({cap}) reached"
        _append_log(alert, {"at": _now().isoformat(), "attempt": alert.attempts,
                            "status": "SKIPPED", "error": alert.last_error})
        db.commit()
        db.refresh(alert)
        return alert
    return _send_now(db, alert, provider or get_sms_provider())


def send_test_sms(db: Session, user: User, *, message: str | None = None,
                  provider: SmsProvider | None = None) -> Alert:
    """Deliver a test SMS to the user's own registered number. Bypasses dedup."""
    if not user.phone_number:
        raise ValueError("no phone number on file for this user")
    body = message or settings.ALERT_TEST_SMS_TEMPLATE.format(
        name=user.name, ts=_now().strftime("%Y-%m-%d %H:%M UTC")
    )
    return dispatch_alert(
        db, message=body, recipient=user.phone_number, recipient_user_id=user.id,
        channel=AlertChannel.SMS, kind="test", force=True, provider=provider,
    )
