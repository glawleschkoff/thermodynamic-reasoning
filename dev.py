"""Dev server with auto-reload.

Usage:  python3 dev.py   (then open http://localhost:8000)

- Change to run.py / pbits/*.py -> run.py runs again (writes docs/data.js), browser reloads
- Change to docs/*.js/html -> browser reloads
The reload code is injected into index.html only when serving; the file itself stays unchanged.
"""
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(ROOT, "docs")
PORT = 8000
CORE = os.path.join(ROOT, "run.py")
WATCH_PY = [os.path.join(ROOT, "run.py")] + [os.path.join(ROOT, "pbits", f) for f in ("model.py", "simulate.py", "bayes.py", "export.py")]
WATCH_VIEWER = ["index.html", "main.js", "data.js"]
# Auto-reload in the browser (polls /__version every 700 ms). Off because it costs memory;
# reload the page manually (Cmd+R) instead. run.py is still re-run automatically.
AUTO_RELOAD = False

state = {"version": 0, "status": "ok"}

SNIPPET = b"""<script>
(function(){let v=null;setInterval(async()=>{try{const r=await (await fetch('/__version',{cache:'no-store'})).json();
if(r.status==='running')return;if(v===null)v=r.version;else if(r.version!==v)location.reload();}catch(e){}},700);})();
</script>
"""


def mtimes(paths):
    out = []
    for p in paths:
        try:
            out.append(os.path.getmtime(p))
        except OSError:
            out.append(0)
    return out


def run_core():
    state["status"] = "running"
    print("[dev] run.py running ...", flush=True)
    env = dict(os.environ, MPLBACKEND="Agg")
    res = subprocess.run([sys.executable, CORE], cwd=ROOT, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("[dev] run.py FEHLER:\n" + res.stderr[-1500:], flush=True)
    else:
        print("[dev] run.py finished, data.js rewritten", flush=True)
    state["status"] = "ok"
    state["version"] += 1


def watcher():
    core_m = mtimes(WATCH_PY)
    view_m = mtimes([os.path.join(VIEWER, f) for f in WATCH_VIEWER])
    while True:
        time.sleep(0.5)
        c = mtimes(WATCH_PY)
        if c != core_m:
            core_m = c
            run_core()
            view_m = mtimes([os.path.join(VIEWER, f) for f in WATCH_VIEWER])  # run.py writes data.js/index.html
            continue
        v = mtimes([os.path.join(VIEWER, f) for f in WATCH_VIEWER])
        if v != view_m:
            view_m = v
            state["version"] += 1
            print("[dev] Viewer-Datei geändert -> Reload", flush=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=VIEWER, **k)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/__version":
            body = ('{"version": %d, "status": "%s"}' % (state["version"], state["status"])).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            with open(os.path.join(VIEWER, "index.html"), "rb") as f:
                html = f.read()
            if AUTO_RELOAD:
                html = html.replace(b"</body>", SNIPPET + b"</body>") if b"</body>" in html else html + SNIPPET
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=watcher, daemon=True).start()
    print(f"[dev] http://localhost:{PORT}  (Ctrl+C to quit)", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
