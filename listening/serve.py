"""Serve a listening test built by listening/make_test.py on localhost, and save the ratings on the server.

Serves --dir (index.html and audio/) on 127.0.0.1:--port only. key.json and config.json are not served, so the page
cannot reveal which letter is which condition. After every rating the page posts all ratings as CSV to /save, and
they are written to <dir>/ovl_ratings.csv, so nothing has to be downloaded. Reach it from another machine with
ssh -L <local port>:localhost:<port> <server>.

    python listening/serve.py --dir outputs/listening/ovl --port 8765
"""

import argparse
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HIDDEN = {"key.json", "config.json"}
HEADER = "page,caption_id,letter,score,played_full\n"
SAVE_LOCK = threading.Lock()  # requests run in threads; two saves at once must not share the temporary file


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if Path(self.translate_path(self.path)).name in HIDDEN:
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if Path(self.translate_path(self.path)).name in HIDDEN:
            self.send_error(404)
            return
        super().do_HEAD()

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
        if not body.startswith(HEADER):
            self.send_error(400)
            return
        out = Path(self.directory) / "ovl_ratings.csv"
        tmp = out.with_suffix(".csv.tmp")
        with SAVE_LOCK:
            tmp.write_text(body)
            tmp.replace(out)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/listening/ovl")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(Path(args.dir).resolve()))
    print(f"Serving {args.dir} on http://127.0.0.1:{args.port}; ratings go to {args.dir}/ovl_ratings.csv", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
