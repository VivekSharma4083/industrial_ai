"""
scripts/verify_live_server.py
=============================
Spins up the Flask backend on localhost, performs live HTTP requests to
GET / and GET /api/health using urllib.request, verifies payload integrity,
and exits cleanly.
"""

import sys
import time
import json
import urllib.request
from pathlib import Path
from threading import Thread

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from werkzeug.serving import make_server
from backend.app import create_app


class ServerThread(Thread):
    def __init__(self, app, host="127.0.0.1", port=5055):
        super().__init__()
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()
        self.port = port
        self.host = host

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def run_live_verification():
    app = create_app()
    port = 5055
    server_thread = ServerThread(app, port=port)
    server_thread.start()
    print(f"[*] Started live test server on http://127.0.0.1:{port}")

    # Give server a moment to bind
    time.sleep(0.5)

    try:
        # 1. Test GET / (Homepage)
        print("[*] Testing GET / ...")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            assert status == 200, f"Expected 200, got {status}"
            assert "INDUSTRIAL AI INSPECTOR" in body, "Homepage title missing"
            print(f"    [+] Root route OK (HTTP {status}, {len(body)} bytes returned)")

        # 2. Test GET /api/health
        print("[*] Testing GET /api/health ...")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
            status = resp.status
            raw_data = resp.read().decode("utf-8")
            data = json.loads(raw_data)
            assert status == 200, f"Expected 200, got {status}"
            assert data["status"] == "healthy", f"Expected healthy, got {data.get('status')}"
            assert data["service"] == "Industrial AI Inspector API"
            print(f"    [+] /api/health OK (HTTP {status})")
            print(f"    [+] Health Payload:\n{json.dumps(data, indent=8)}")

        # 3. Test static file serving /css/styles.css
        print("[*] Testing GET /css/styles.css ...")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/css/styles.css", timeout=5) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            assert status == 200
            assert "--bg-primary" in body
            print(f"    [+] Static CSS route OK (HTTP {status})")

        # 4. Test static file serving /js/app.js
        print("[*] Testing GET /js/app.js ...")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/js/app.js", timeout=5) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            assert status == 200
            assert "checkHealth" in body
            print(f"    [+] Static JS route OK (HTTP {status})")

        print("\n[SUCCESS] Live HTTP server verification passed completely!")
        return 0

    except Exception as exc:
        print(f"\n[FAILURE] Live verification failed: {exc}")
        return 1
    finally:
        print("[*] Shutting down live test server...")
        server_thread.shutdown()
        server_thread.join()
        print("[*] Server shutdown complete.")


if __name__ == "__main__":
    sys.exit(run_live_verification())
