"""Receiver for h2_beta.html ?dump=NAME: writes NAME.meta/.psi/.lab/.K into DIR. usage: python3 tools/dump_server.py DIR
Port 8767 (8766 is taken by another local static server). For tools/check_energy.py."""
import http.server, urllib.parse, os, sys
OUT = sys.argv[1]
class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
    def do_OPTIONS(self): self.send_response(204); self._cors(); self.end_headers()
    def do_POST(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        n = int(self.headers.get('Content-Length', 0)); body = self.rfile.read(n)
        name = os.path.basename(q.get('name', ['x'])[0]); part = os.path.basename(q.get('part', ['p'])[0])
        with open(os.path.join(OUT, '%s.%s' % (name, part)), 'wb') as f: f.write(body)
        print('got', name, part, n, flush=True)
        self.send_response(200); self._cors(); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', 8767), H).serve_forever()
