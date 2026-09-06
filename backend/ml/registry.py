"""
Lazy, instrumented model registry.

Adapted from ``poseatsea/registry.py`` (the merge decision: adopt PS's registry
pattern for model lifecycle - docs/MERGE_ARCHITECTURE.md).

Each model loads on first use, once per process, behind a lock; load time and
parameter count are recorded so the System page can show what is resident. A
session that never touches the AIS models never pays for torch.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.core.logging import get_logger

log = get_logger("backend.ml.registry")


@dataclass
class _Handle:
    key: str
    display_name: str
    loader: Callable[[], Any]
    _obj: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    load_seconds: float | None = None
    param_count: int | None = None
    error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._obj is not None

    def get(self) -> Any:
        if self._obj is not None:
            return self._obj
        with self._lock:
            if self._obj is not None:
                return self._obj
            log.info("loading model %r ...", self.key)
            start = time.perf_counter()
            try:
                obj = self.loader()
            except Exception as exc:  # noqa: BLE001 - recorded and re-raised
                self.error = f"{type(exc).__name__}: {exc}"
                log.exception("failed to load model %r", self.key)
                raise
            self.load_seconds = time.perf_counter() - start
            self.param_count = _count_params(obj)
            self._obj = obj
            log.info("loaded %r in %.2fs (%s params)", self.key, self.load_seconds,
                     f"{self.param_count:,}" if self.param_count else "n/a")
            return self._obj

    def unload(self) -> None:
        with self._lock:
            self._obj = None
            self.load_seconds = self.param_count = None

    def status(self) -> dict:
        return {
            "key": self.key, "name": self.display_name, "loaded": self.is_loaded,
            "load_seconds": round(self.load_seconds, 3) if self.load_seconds else None,
            "parameters": self.param_count, "error": self.error,
        }


def _count_params(obj: Any) -> int | None:
    target = obj
    if isinstance(obj, tuple):
        for item in obj:
            if hasattr(item, "parameters"):
                target = item
                break
    params = getattr(target, "parameters", None)
    if callable(params):
        try:
            return int(sum(p.numel() for p in params()))
        except Exception:  # noqa: BLE001
            return None
    return None


class ModelRegistry:
    _instance: "ModelRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._handles: dict[str, _Handle] = {}
        self._register_defaults()

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _register_defaults(self) -> None:
        def _load_ais():
            from backend.ml.ais.anomaly import load_model
            return load_model()                     # (model, scaler)

        def _load_traj():
            from backend.ml.trajectory.lstm import load_model
            return load_model()

        def _load_sar():
            from backend.ml.sar.classifier import get_classifier
            return get_classifier("SAR")

        self._handles = {
            "ais_anomaly": _Handle("ais_anomaly", "AIS Anomaly Autoencoder (+ StandardScaler)", _load_ais),
            "trajectory": _Handle("trajectory", "Trajectory LSTM", _load_traj),
            "sar_classifier": _Handle("sar_classifier", "SAR Oil/Look-alike RF+GB Ensemble", _load_sar),
        }

    def __getitem__(self, key: str) -> _Handle:
        return self._handles[key]

    def status(self) -> dict:
        return {"models": [h.status() for h in self._handles.values()]}

    def unload_all(self) -> None:
        for h in self._handles.values():
            h.unload()


def get_registry() -> ModelRegistry:
    return ModelRegistry.instance()


def get_ais_anomaly_model():
    """``(autoencoder, StandardScaler)`` - loaded once, scaler always present."""
    return get_registry()["ais_anomaly"].get()


def get_trajectory_model():
    return get_registry()["trajectory"].get()
