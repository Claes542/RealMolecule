"""
Light-nucleus binding ladder, run headless.

The RealNucleus cases live as browser pages (nucleus_*.html) driven by
molecule_nucleus.js, whose relaxation loop is pumped by p5's draw(). This runs
them without a person watching: for each case it extracts the configuration
block from the original page, wraps it in a minimal driver that posts back the
converged energy instead of drawing, and drives Chrome over a local server.

Headless will not do. molecule_nucleus.js is a WebGPU solver and headless Chrome
never resolves an adapter (gpuReady stays false), so Chrome runs with a real
window -- placed offscreen, in a throwaway profile -- and the page reports its
result back to the server this script is running.

The originals are never modified and never need to be.

Only RATIOS are meaningful: the absolute MeV scale is the model's open problem,
so every energy is divided by the deuteron's and compared with the measured
binding-energy ratio.

usage: python3 run_ladder.py [--grid N] [--timeout S] [case ...]
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path('/Users/cgjoh/Development/H2O')
HERE = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

# case -> (page, measured binding energy in MeV)
# AME2020 values; the deuteron is the reference for every ratio.
CASES = {
    'deuteron': ('nucleus_deuterium_2shell.html', 2.2246),
    'he3':      ('nucleus_he3.html',              7.7180),
    'he4':      ('nucleus_he4_2plus2.html',      28.2957),
    'li6':      ('nucleus_li6_matched.html',     31.9946),
    'be8':      ('nucleus_be8_asym.html',        56.4996),
}

DRIVER = """<!DOCTYPE html><html><body style="margin:0;background:#000">
<pre id="out" style="color:#0f0;font:12px monospace">starting</pre>
<script>
window.__err = [];
window.onerror = function (m, src, l) { window.__err.push(m + ' @' + (src||'').split('/').pop() + ':' + l);
  var e = document.getElementById('out'); if (e) e.textContent = 'ERROR ' + window.__err.join(' | '); };
</script>
<script>
%(config)s
window.__label = %(label)r;
window.__done = false;
window.onSweepDone = function (E) {
  if (window.__done) return;
  window.__done = true;
  var r = JSON.stringify({case: window.__label, E: Number(E), trace: window.__trace || []});
  document.title = 'RESULT ' + r;
  document.getElementById('out').textContent = 'RESULT ' + r;
  fetch('/__result?payload=' + encodeURIComponent(r));
};
// progress into the DOM so a --dump-dom shows how far it got even without convergence
window.__trace = [];
setInterval(function () {
  if (window.__done) return;
  if (window._prevE !== undefined && isFinite(window._prevE)) {
    var t = window.__trace;
    if (!t.length || Math.abs(t[t.length-1] - window._prevE) > 1e-9) t.push(Number(Number(window._prevE).toFixed(4)));
  }
  var e = document.getElementById('out');
  var g = 'gpu?';
  try { g = 'gpuReady=' + gpuReady + ' gpuError=' + (gpuError || 'none') + ' init=' + (typeof initProgress !== 'undefined' ? initProgress : '?'); }
  catch (ex) { g = 'gpu-binding: ' + ex.message; }
  if (e) e.textContent = 'PROGRESS E=' + (window._prevE === undefined ? 'n/a' : Number(window._prevE).toFixed(6))
                       + ' conv=' + (window._convCount || 0) + ' | ' + g;
}, 250);
</script>
<script src="/molecule_nucleus.js"></script>
<script src="/real_nucleus_numerics/p5.min.js"></script>
</body></html>
"""


def build(label, page, grid, thresh=None):
    html = (ROOT / page).read_text()
    body = html[html.index('<script>') + 8:html.rindex('</script>')]
    cfg = body[:body.index('var s = document.createElement')]
    # strip the page's own title-polling and its onSweepDone, keep the USER_* block
    cfg = re.sub(r'window\.onSweepDone\s*=\s*function[\s\S]*?\};', '', cfg)
    if thresh is not None:
        cfg += f'\nwindow.CONVERGENCE_THRESHOLD = {thresh};\n'
    if grid:
        # the page computes atom indices from its own gridN, so a bare USER_NN
        # override would leave the seeds outside the box: rescale the page's
        # gridN instead, before the indices are formed.
        cfg = re.sub(r'var gridN = \d+', f'var gridN = {grid}', cfg, count=1)
        cfg = re.sub(r'N2 = \d+', f'N2 = {grid // 2}', cfg, count=1)
    out = HERE / f'_driver_{label}.html'
    out.write_text(DRIVER % dict(config=cfg, label=label))
    return out


class _Handler(SimpleHTTPRequestHandler):
    """Serves the repo, and collects one result per case at /__result."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/__result'):
            q = parse_qs(urlparse(self.path).query)
            self.server.result = json.loads(q['payload'][0])
            self.send_response(204); self.end_headers()
            return
        return super().do_GET()


def serve():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0),
                                partial(_Handler, directory=str(ROOT)))
    httpd.result = None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(httpd, driver, timeout_s):
    """Open the driver in a real Chrome window (offscreen) and wait for its post-back."""
    httpd.result = None
    url = f'http://127.0.0.1:{httpd.server_address[1]}/real_nucleus_numerics/{driver.name}'
    profile = tempfile.mkdtemp(prefix='nucladder-')
    chrome = subprocess.Popen(
        [CHROME, '--new-window', f'--user-data-dir={profile}',
         '--no-first-run', '--no-default-browser-check',
         '--window-position=3000,3000', '--window-size=500,400',
         '--disable-features=CalculateNativeWinOcclusion',
         '--autoplay-policy=no-user-gesture-required', url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while httpd.result is None and time.time() - t0 < timeout_s:
            time.sleep(2)
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=15)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
    return httpd.result


def main():
    args = sys.argv[1:]
    grid = None
    timeout_s = 1800
    if '--grid' in args:
        i = args.index('--grid'); grid = int(args[i + 1]); del args[i:i + 2]
    thresh = None
    if '--thresh' in args:
        i = args.index('--thresh'); thresh = float(args[i + 1]); del args[i:i + 2]
    if '--timeout' in args:
        i = args.index('--timeout'); timeout_s = int(args[i + 1]); del args[i:i + 2]
    want = args or list(CASES)

    if not (HERE / 'p5.min.js').exists():
        subprocess.run(['curl', '-sSo', str(HERE / 'p5.min.js'),
                        'https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js'],
                       check=True)

    httpd = serve()
    results = {}
    for label in want:
        page, exp = CASES[label]
        drv = build(label, page, grid, thresh)
        print(f'--- {label} ({page}, grid {grid or "as configured"}) ...', flush=True)
        t0 = time.time()
        r = run(httpd, drv, timeout_s)
        if r is None:
            print(f'    no result in {timeout_s}s', flush=True)
            continue
        results[label] = r['E']
        tr = r.get('trace') or []
        tail = ' '.join(f'{v:.3f}' for v in tr[-6:])
        print(f'    E = {r["E"]:.6f}   ({time.time() - t0:.0f}s)  samples={len(tr)}  tail: {tail}', flush=True)

    if 'deuteron' not in results:
        print('\nno deuteron energy: ratios need it as the reference')
        print(json.dumps(results, indent=2))
        return
    Ed = results['deuteron']
    print(f'\n{"case":>9} {"E (code)":>12} {"E/E_d":>9} {"exp ratio":>10} {"dev":>8}')
    rows = []
    for label in CASES:
        if label not in results:
            continue
        E = results[label]
        ratio = E / Ed
        exp = CASES[label][1] / CASES['deuteron'][1]
        dev = 100 * (ratio - exp) / exp
        rows.append(dict(case=label, E=round(E, 6), ratio=round(ratio, 2),
                         exp=round(exp, 2), dev_pct=round(dev, 1)))
        print(f'{label:>9} {E:>12.4f} {ratio:>9.2f} {exp:>10.2f} {dev:>7.1f}%')
    (HERE / 'ladder_results.json').write_text(json.dumps(rows, indent=2))
    print(f'\nwritten to {HERE / "ladder_results.json"}')


if __name__ == '__main__':
    main()
