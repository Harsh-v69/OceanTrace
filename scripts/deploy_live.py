#!/usr/bin/env python
"""
Temporary public deployment of the OceanTrace prototype.

Runs a live-reloading uvicorn server on :8000 and opens a public HTTPS tunnel
to it with ngrok, so edits to ``backend/`` show up instantly at a URL anyone
can open.

    python scripts/deploy_live.py

One-time ngrok setup (free account at https://dashboard.ngrok.com):

    ngrok config add-authtoken <YOUR_TOKEN>
        - or set it for this run only -
    Windows :  set NGROK_AUTHTOKEN=<YOUR_TOKEN>
    bash    :  export NGROK_AUTHTOKEN=<YOUR_TOKEN>

Press Ctrl+C to close the tunnel and stop the server.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))
_IS_WIN = os.name == "nt"


def _print_box(public_url: str) -> None:
    bar = "=" * 74
    lines = [
        "",
        bar,
        "   OceanTrace is LIVE  -  temporary public tunnel (ngrok)",
        bar,
        f"   PUBLIC URL   ->  {public_url}",
        f"   Console      ->  {public_url}/app/",
        f"   API docs     ->  {public_url}/api/docs",
        f"   Local        ->  http://{HOST}:{PORT}/app/",
        bar,
        "   Seeded sign-ins  (password = DEFAULT_USER_PASSWORD env var):",
        "     national@oceantrace.gov.in  /  regional@oceantrace.gov.in  /  pilot@oceantrace.gov.in",
        bar,
        "   Code reloads automatically on save. Ctrl+C stops everything.",
        bar,
        "",
    ]
    print("\n".join(lines), flush=True)


def _wait_for_port(proc: subprocess.Popen, timeout: float = 40.0) -> bool:
    """Block until uvicorn is accepting connections (or it dies / times out)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((HOST, PORT)) == 0:
                return True
        time.sleep(0.5)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if _IS_WIN:
        # --reload spawns a child worker; kill the whole tree.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, check=False,
        )
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    token = os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_AUTH_TOKEN")

    try:
        from pyngrok import conf, ngrok
        from pyngrok.exception import PyngrokError
    except ImportError:
        print("pyngrok is not installed.  ->  pip install pyngrok", file=sys.stderr)
        return 1

    if token:
        conf.get_default().auth_token = token

    # 1) start the live-reloading API (watch only backend/, not .venv)
    cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", HOST, "--port", str(PORT),
        "--reload", "--reload-dir", "backend",
    ]
    print(f"starting: {' '.join(cmd)}", flush=True)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if _IS_WIN else 0
    server = subprocess.Popen(
        cmd, cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        creationflags=creationflags,
    )

    if not _wait_for_port(server):
        print("\nuvicorn did not come up - see its output above.", file=sys.stderr)
        _stop_server(server)
        return server.returncode or 1

    # 2) open the public HTTPS tunnel (ngrok v3 is HTTPS by default)
    try:
        tunnel = ngrok.connect(PORT, "http")
    except PyngrokError as exc:
        print(
            "\nngrok could not start:\n"
            f"  {exc}\n\n"
            "Add a free authtoken and re-run:\n"
            "  ngrok config add-authtoken <YOUR_TOKEN>\n"
            "  (get it at https://dashboard.ngrok.com/get-started/your-authtoken)\n",
            file=sys.stderr,
        )
        _stop_server(server)
        return 1

    public_url = tunnel.public_url
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[len("http://"):]
    _print_box(public_url)

    # 3) hold open until Ctrl+C or the server exits
    try:
        while True:
            if server.poll() is not None:
                print("\nuvicorn exited - closing the tunnel.", flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C received - shutting down...", flush=True)
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
        except Exception:  # noqa: BLE001
            pass
        try:
            ngrok.kill()
        except Exception:  # noqa: BLE001
            pass
        _stop_server(server)
        print("stopped.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
