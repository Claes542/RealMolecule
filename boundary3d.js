// boundary3d.js -- 3D view of molecule.js's free domain boundaries.
//
// Adds a "3D boundary" button. On click it reads the label grid and the amplitude from the GPU
// (window.readLabelsPsi, molecule.js) and draws every cell face where the owning domain changes,
// keeping only faces where the density on either side exceeds a cutoff (otherwise each wall runs
// out to the box). The surface is deliberately NOT smoothed: the staircase is the boundary the
// solver actually uses. Faces against the hard core (labels >= NELEC) are drawn grey.
// Colours are per domain PAIR; nuclei are small spheres from window.USER_ATOMS.
(function () {
  var THREE_URL = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  var ORBIT_URL = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';

  function load(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  // Pure mesh extraction (also used headless for testing).
  // Returns { groups: { key: Float32Array of triangle positions } , counts }
  function extract(d, cutoff) {
    var S = d.S, h = d.h, lab = d.lab, psi = d.psi, NE = d.NELEC, S2 = S * S, c = (S - 1) / 2;
    var groups = {};
    function key(a, b) {
      var A = a >= NE ? 'core' : a, B = b >= NE ? 'core' : b;
      if (A === B) return null;
      if (A === 'core') return 'core|' + B;
      if (B === 'core') return 'core|' + A;
      return Math.min(A, B) + '|' + Math.max(A, B);
    }
    function push(k, q) { (groups[k] || (groups[k] = [])).push.apply(groups[k], q); }
    var strides = [S2, S, 1];
    for (var i = 0; i < S; i++) for (var j = 0; j < S; j++) for (var k = 0; k < S; k++) {
      var id = i * S2 + j * S + k, la = lab[id];
      var ra = psi[id] * psi[id];
      var ijk = [i, j, k];
      for (var ax = 0; ax < 3; ax++) {
        if (ijk[ax] === S - 1) continue;
        var nb = id + strides[ax], lb = lab[nb];
        if (lb === la) continue;
        var rb = psi[nb] * psi[nb];
        if (Math.max(ra, rb) < cutoff) continue;
        var kk = key(la, lb); if (!kk) continue;
        // face at ijk[ax] + 0.5 along ax, spanning the cell in the other two axes
        var p = [(i - c) * h, (j - c) * h, (k - c) * h];
        p[ax] += 0.5 * h;
        var u = (ax + 1) % 3, v = (ax + 2) % 3, hu = 0.5 * h;
        var corners = [[-hu, -hu], [hu, -hu], [hu, hu], [-hu, hu]].map(function (o) {
          var q = p.slice(); q[u] += o[0]; q[v] += o[1]; return q;
        });
        push(kk, [].concat(corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]));
      }
    }
    var out = {}, counts = {};
    Object.keys(groups).forEach(function (g) { out[g] = new Float32Array(groups[g]); counts[g] = groups[g].length / 18; });
    return { groups: out, counts: counts };
  }
  window.boundary3dExtract = extract;

  var PALETTE = [0x4e79a7, 0xf28e2b, 0x59a14f, 0xe15759, 0xb07aa1, 0x76b7b2, 0xedc948, 0xff9da7, 0x9c755f];

  async function open() {
    if (!window.readLabelsPsi) { alert('molecule.js not ready (readLabelsPsi missing)'); return; }
    if (!window.THREE) { await load(THREE_URL); await load(ORBIT_URL); }
    var d = await window.readLabelsPsi();
    var ov = document.getElementById('b3d');
    if (ov) ov.remove();
    ov = document.createElement('div'); ov.id = 'b3d';
    ov.style.cssText = 'position:fixed;inset:0;background:#0b0b10;z-index:1000;';
    ov.innerHTML = '<div style="position:absolute;top:10px;left:10px;color:#ddd;font:12px monospace;z-index:2;'
      + 'background:rgba(0,0,0,.6);padding:8px;border-radius:6px;max-width:360px">'
      + '<b>Free domain boundaries</b> (cell faces where the owner changes)<br>'
      + 'density cutoff 10^<input id="b3dc" type="range" min="-5" max="-1" step="0.5" value="-3" style="vertical-align:middle"> '
      + '<span id="b3dcv">-3</span><br><span id="b3dleg"></span><br>'
      + '<button id="b3dr">refresh</button> <button id="b3dx">close</button> <span style="color:#888">drag to rotate, wheel to zoom</span></div>';
    document.body.appendChild(ov);
    var W = window.innerWidth, H = window.innerHeight;
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H); renderer.setPixelRatio(window.devicePixelRatio || 1);
    ov.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    var cam = new THREE.PerspectiveCamera(40, W / H, 0.05, 500);
    var span = d.S * d.h;
    cam.position.set(span * 0.35, span * 0.25, span * 0.45);
    var ctl = new THREE.OrbitControls(cam, renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1, 2, 3); scene.add(dl);
    var dl2 = new THREE.DirectionalLight(0xffffff, 0.3); dl2.position.set(-2, -1, -1); scene.add(dl2);

    // nuclei
    var c = (d.S - 1) / 2;
    (window.USER_ATOMS || []).forEach(function (a) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffffff }));
      m.position.set((a.i - c) * d.h, (a.j - c) * d.h, (a.k - c) * d.h); scene.add(m);
    });

    var meshes = [];
    function build(cut) {
      meshes.forEach(function (m) { scene.remove(m); m.geometry.dispose(); }); meshes = [];
      var ex = extract(d, cut), leg = [], n = 0;
      var names = (window.USER_ATOMS || []).map(function (a) { return a.el; });
      Object.keys(ex.groups).sort().forEach(function (g) {
        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(ex.groups[g], 3));
        geo.computeVertexNormals();
        var isCore = g.indexOf('core') === 0;
        var col = isCore ? 0x888888 : PALETTE[n++ % PALETTE.length];
        var mat = new THREE.MeshStandardMaterial({ color: col, transparent: true, opacity: isCore ? 0.35 : 0.6,
          side: THREE.DoubleSide, depthWrite: false });
        var m = new THREE.Mesh(geo, mat); scene.add(m); meshes.push(m);
        var lbl = g.split('|').map(function (x) { return x === 'core' ? 'core' : (names[+x] || ('d' + x)); }).join(' | ');
        leg.push('<span style="color:#' + col.toString(16).padStart(6, '0') + '">&#9632;</span> ' + lbl
                 + ' <span style="color:#888">(' + ex.counts[g] + ' faces)</span>');
      });
      document.getElementById('b3dleg').innerHTML = leg.join('<br>');
    }
    var cutEl = document.getElementById('b3dc');
    cutEl.oninput = function () { document.getElementById('b3dcv').textContent = cutEl.value; build(Math.pow(10, +cutEl.value)); };
    document.getElementById('b3dr').onclick = async function () { d = await window.readLabelsPsi(); build(Math.pow(10, +cutEl.value)); };
    var alive = true;
    document.getElementById('b3dx').onclick = function () { alive = false; renderer.dispose(); ov.remove(); };
    build(1e-3);
    (function loop() { if (!alive) return; ctl.update(); renderer.render(scene, cam); requestAnimationFrame(loop); })();
  }
  window.openBoundary3D = open;

  if (typeof document !== 'undefined' && document.body) {
    var b = document.createElement('button');
    b.textContent = '3D boundary';
    b.style.cssText = 'position:fixed;bottom:12px;right:12px;z-index:999;padding:6px 10px;font:12px monospace;'
      + 'background:#223;color:#cdf;border:1px solid #446;border-radius:5px;cursor:pointer';
    b.onclick = open;
    document.body.appendChild(b);
  }
})();
