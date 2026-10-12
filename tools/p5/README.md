# Running the p5.js sketches headless

The published RealQM results on the blog were produced by individual p5.js sketches in the
web editor, one run per species. These scripts fetch a sketch and drive it to its own
stopping point so the numbers can be reproduced and checked for convergence.

## Why bother

Because it answers questions no reimplementation can. Running the helium sketch
(`UXZ8eA5WM`) to its own `t<3000` limit shows the energy rising monotonically through
the quoted `-2.903` at about `t~2000` and reaching `-2.826` by `t=3000`, still rising --
so that value is one the relaxation *transits*, not one it converges to. Only the original
program can establish that.

## Usage

    # 1. fetch the source (the editor's API returns JSON with files[].content)
    curl -s https://editor.p5js.org/editor/claesjohnson/projects/<id> \
      | python3 -c "import json,sys; d=json.load(sys.stdin); \
          print([f['content'] for f in d['files'] if f['name']=='sketch.js'][0])" \
      > sk_<id>.js

    # 2. wrap it in a page that loads p5 and exposes a reader
    python3 mkrunner.py <id>

    # 3. serve and drive
    python3 -m http.server 8001 &
    python3 drive_sketch.py <id> 1500

## Traps, all of which cost time once

- **Top-level `let` in a `<script>` is script-scoped, not on `window`.** You cannot read the
  sketch's `E` or `Etot` with `Runtime.evaluate` from outside. `mkrunner.py` appends a reader
  function *inside the same script block* so it closes over the bindings.
- **Variable names differ between sketches.** The helium sketch reports `E`; the ionization
  sketches use `Ekin`/`Epot`/`Etot`. Grep for `text(` to see what a sketch actually displays.
- **Node is not installed here.** `p5stub.js` is a stub for the Node route and is unused;
  the browser route is also more faithful, since it runs in the real p5 environment.
- **Chrome can have the CDP port open with zero page targets.** Create one with
  `curl -X PUT "http://localhost:9222/json/new?<url>"`.
- Energies are drawn with `text()` onto the canvas, so screenshots are not a readout --
  the closure reader is the only reliable route.
