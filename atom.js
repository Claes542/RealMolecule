// atom.js — Shell-based atomic solver
// Per-electron arrays: U[m], W[m], P[m]
// Weighted Laplacian div(w·∇u) + front tracking + simultaneous Poisson
//
// Configuration via window globals (set by HTML page):
//   USER_Z_KERNEL  — nuclear charge (e.g., 2 for He)
//   USER_SHELLS    — array of { ne: electrons, Rmax: init outer radius, Zeff: init Zeff }
//   USER_NN        — grid size (default 100)
//   USER_SCREEN    — box size in au (default 10)
//   USER_STEPS     — max steps (default 200)
//   USER_D         — diffusion parameter (default 0.1)

console.log("atom.js loaded");

const Z_kernel = window.USER_Z_KERNEL || 2;
const shellConfig = window.USER_SHELLS || [
  { ne: 1, Rmax: 1.0, Zeff: 2 },
  { ne: 1, Rmax: 5.0, Zeff: 1 }
];
const nShells = shellConfig.length;
const NN = window.USER_NN || 100;
const scr = window.USER_SCREEN || 10;
const maxSteps = window.USER_STEPS || 200;
const d = window.USER_D || 0.1;

const h = scr / NN;
const h2 = h * h;
const h3 = h * h * h;
const N2 = Math.floor(NN / 2);
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const dt = d * h2;
const poissonMode = window.USER_POISSON || 'multigrid';

// Per-shell arrays
const U = [], W = [], P = [];
const K = new Float32Array(S3);
const c = new Float32Array(nShells);

for (let m = 0; m < nShells; m++) {
  U.push(new Float32Array(S3));
  W.push(new Float32Array(S3));
  P.push(new Float32Array(S3));
}

function idx(i, j, k) { return i * S2 + j * S + k; }

// Initialize
console.log('atom.js: Z=' + Z_kernel + ', ' + nShells + ' shells, NN=' + NN + ', scr=' + scr);

// Compute shell boundaries from config
// Shell 0: 0 < r < Rmax[0]
// Shell m: Rmax[m-1] < r < Rmax[m]
const Rmin = [0];
const Rmax = [shellConfig[0].Rmax];
for (let m = 1; m < nShells; m++) {
  Rmin.push(shellConfig[m - 1].Rmax);
  Rmax.push(shellConfig[m].Rmax || scr / 2);
}

for (let i = 0; i <= NN; i++) {
  for (let j = 0; j <= NN; j++) {
    for (let k = 0; k <= NN; k++) {
      const id = idx(i, j, k);
      const r2 = (i - N2) * (i - N2) * h2 + (j - N2) * (j - N2) * h2 + (k - N2) * (k - N2) * h2 + 0.12 * h2;
      const r = Math.sqrt(r2);

      K[id] = Z_kernel / r;

      // Initialize U and W per shell
      for (let m = 0; m < nShells; m++) {
        const cfg = shellConfig[m];
        let inShell;
        if (cfg.split === 'halfspace') {
          // Half-space: divide by x-plane, no radial limit
          if (i === N2) {
            // Midplane: both halves get W=0.5 for symmetry
            inShell = true;
            W[m][id] = 0.5;
          } else {
            inShell = (cfg.halfIndex === 0) ? (i < N2) : (i > N2);
          }
        } else if (cfg.outerHalf) {
          // Radial shell but only in half-space (x > center)
          inShell = (r >= Rmin[m] && r < Rmax[m] && i > N2);
        } else {
          // Radial shells with 1-cell smooth boundary
          inShell = (r >= Rmin[m] && r < Rmax[m]);
        }
        if (inShell) {
          U[m][id] = Math.exp(-cfg.Zeff * r);
          // Smooth W at shell boundaries: linear ramp over 2 grid cells
          let wVal = 1.0;
          const smoothW = 2 * h;
          if (Rmin[m] > 0) {
            const distFromInner = r - Rmin[m];
            if (distFromInner < smoothW) wVal = Math.min(wVal, distFromInner / smoothW);
          }
          const distFromOuter = Rmax[m] - r;
          if (distFromOuter < smoothW) wVal = Math.min(wVal, distFromOuter / smoothW);
          wVal = Math.max(0.01, wVal);  // never fully zero inside
          if (W[m][id] === 0) W[m][id] = wVal;  // don't overwrite midplane W=0.5
        }
      }

      // Initialize P: Coulomb potential from other electrons
      // Approximate as 0.5/r for all (Poisson will converge to correct values)
      for (let m = 0; m < nShells; m++) {
        P[m][id] = 0.5 / r;
      }
    }
  }
}

// Normalize initial wavefunctions
for (let m = 0; m < nShells; m++) {
  let n2 = 0;
  for (let id = 0; id < S3; id++) n2 += U[m][id] * U[m][id] * h3;
  const sc = 1 / Math.sqrt(n2);
  for (let id = 0; id < S3; id++) U[m][id] *= sc;
  console.log('Shell ' + m + ': ne=' + shellConfig[m].ne + ' R=[' + Rmin[m].toFixed(2) + ',' + Rmax[m].toFixed(2) + '] Zeff=' + shellConfig[m].Zeff + ' norm=' + (n2 * sc * sc).toFixed(4));
}

// Multigrid V-cycle Poisson solver: ∇²P = -rhs
// Jacobi smoother + restrict + coarse solve + prolongate
function jacobiSmooth(P, rhs, N, h2, nIter) {
  const S = N + 1, S2 = S * S;
  for (let iter = 0; iter < nIter; iter++) {
    for (let i = 1; i < N; i++)
      for (let j = 1; j < N; j++)
        for (let k = 1; k < N; k++) {
          const id = i * S2 + j * S + k;
          P[id] = (P[id + S2] + P[id - S2] + P[id + S] + P[id - S] + P[id + 1] + P[id - 1] + h2 * rhs[id]) / 6;
        }
  }
}

function restrict(fine, Nf) {
  const Nc = Math.floor(Nf / 2);
  const Sf = Nf + 1, Sf2 = Sf * Sf;
  const Sc = Nc + 1, Sc2 = Sc * Sc;
  const coarse = new Float32Array(Sc * Sc * Sc);
  for (let i = 1; i < Nc; i++)
    for (let j = 1; j < Nc; j++)
      for (let k = 1; k < Nc; k++) {
        const fi = 2 * i, fj = 2 * j, fk = 2 * k;
        coarse[i * Sc2 + j * Sc + k] = fine[fi * Sf2 + fj * Sf + fk];
      }
  return { data: coarse, N: Nc, S: Sc, S2: Sc2 };
}

function prolongate(coarse, Nc, fine, Nf) {
  const Sf = Nf + 1, Sf2 = Sf * Sf;
  const Sc = Nc + 1, Sc2 = Sc * Sc;
  for (let i = 1; i < Nc; i++)
    for (let j = 1; j < Nc; j++)
      for (let k = 1; k < Nc; k++) {
        const val = coarse[i * Sc2 + j * Sc + k];
        const fi = 2 * i, fj = 2 * j, fk = 2 * k;
        // Inject to fine grid (nearest neighbor)
        for (let di = 0; di <= 1; di++)
          for (let dj = 0; dj <= 1; dj++)
            for (let dk = 0; dk <= 1; dk++) {
              const fid = (fi + di) * Sf2 + (fj + dj) * Sf + (fk + dk);
              if (fi + di > 0 && fi + di < Nf && fj + dj > 0 && fj + dj < Nf && fk + dk > 0 && fk + dk < Nf)
                fine[fid] += val;
            }
      }
}

function computeResidual(P, rhs, N, h2) {
  const S = N + 1, S2 = S * S;
  const res = new Float32Array(S * S * S);
  for (let i = 1; i < N; i++)
    for (let j = 1; j < N; j++)
      for (let k = 1; k < N; k++) {
        const id = i * S2 + j * S + k;
        const lapP = (P[id + S2] + P[id - S2] + P[id + S] + P[id - S] + P[id + 1] + P[id - 1] - 6 * P[id]) / h2;
        res[id] = rhs[id] + lapP;  // residual = rhs - (-∇²P) = rhs + ∇²P
      }
  return res;
}

function poissonVcycle(P, rhs) {
  const N1 = NN, h1 = h, h1_2 = h2;

  // Pre-smooth
  jacobiSmooth(P, rhs, N1, h1_2, 4);

  // Compute residual
  const res1 = computeResidual(P, rhs, N1, h1_2);

  // Restrict to coarse
  const c = restrict(res1, N1);
  const Nc = c.N;
  const hc = h1 * 2, hc2 = hc * hc;

  // Coarse solve (many Jacobi iterations)
  const Pc = new Float32Array(c.S * c.S * c.S);
  jacobiSmooth(Pc, c.data, Nc, hc2, 20);

  // Prolongate correction
  prolongate(Pc, Nc, P, N1);

  // Post-smooth
  jacobiSmooth(P, rhs, N1, h1_2, 4);
}

// Physics step — direct port of working p5.js code
function physicsStep() {
  let E = 0, E_T = 0, E_eK = 0, E_ee = 0;

  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = idx(i, j, k);

        // Front tracking velocity c[m] — exact p5.js formula
        for (let m = 0; m < nShells; m++) {
          c[m] = 0;
          for (let n = 0; n < nShells; n++) {
            if (n !== m) c[m] -= U[n][id];
            c[m] = 0.5 * (c[m] + U[m][id]);
          }
        }

        const ip = idx(i+1,j,k), im = idx(i-1,j,k);
        const jp = idx(i,j+1,k), jm = idx(i,j-1,k);
        const kp = idx(i,j,k+1), km = idx(i,j,k-1);

        for (let m = 0; m < nShells; m++) {
          // Front tracking: dw/dt = visc*Lap(w) + speed*|grad(w)|
          // Skip if shell has fixed boundary (e.g., half-space split)
          if (!shellConfig[m].fixedW) {
            const lapW = (W[m][ip] + W[m][im] + W[m][jp] + W[m][jm] + W[m][kp] + W[m][km] - 6 * W[m][id]) / h2;
            const gwx = (W[m][ip] - W[m][im]) / h;
            const gwy = (W[m][jp] - W[m][jm]) / h;
            const gwz = (W[m][kp] - W[m][km]) / h;
            const gradW = Math.sqrt(gwx * gwx + gwy * gwy + gwz * gwz);
            W[m][id] += 2 * dt * Math.abs(c[m]) * lapW + 9 * dt * c[m] * gradW;
          }

          // Weighted Laplacian: div(w · ∇u)
          const wlap =
            0.5 * d * ((U[m][ip] - U[m][id]) * (W[m][ip] + W[m][id]) * 0.5
                      -(U[m][id] - U[m][im]) * (W[m][id] + W[m][im]) * 0.5)
          + 0.5 * d * ((U[m][jp] - U[m][id]) * (W[m][jp] + W[m][id]) * 0.5
                      -(U[m][id] - U[m][jm]) * (W[m][id] + W[m][jm]) * 0.5)
          + 0.5 * d * ((U[m][kp] - U[m][id]) * (W[m][kp] + W[m][id]) * 0.5
                      -(U[m][id] - U[m][km]) * (W[m][id] + W[m][km]) * 0.5);

          U[m][id] += wlap + dt * (K[id] - 2 * P[m][id]) * U[m][id] * W[m][id];

          // Energy: only count where this electron's W dominates
          let dominated = true;
          for (let n = 0; n < nShells; n++) {
            if (n !== m && W[n][id] >= W[m][id]) { dominated = false; break; }
          }
          if (dominated && W[m][id] > 0.01) {
            // Gradient: only use neighbor if it's also in this electron's domain (W[m] > W[n])
            let domIp = true, domJp = true, domKp = true;
            for (let n = 0; n < nShells; n++) {
              if (n !== m) {
                if (W[n][ip] >= W[m][ip]) domIp = false;
                if (W[n][jp] >= W[m][jp]) domJp = false;
                if (W[n][kp] >= W[m][kp]) domKp = false;
              }
            }
            const gx = domIp ? U[m][ip] - U[m][id] : 0;
            const gy = domJp ? U[m][jp] - U[m][id] : 0;
            const gz = domKp ? U[m][kp] - U[m][id] : 0;
            E_T += 0.5 * (gx * gx + gy * gy + gz * gz) * h;
            E_ee += P[m][id] * U[m][id] * U[m][id] * h3;
            E_eK -= K[id] * U[m][id] * U[m][id] * h3;
          }

          // Poisson update — simultaneous (inside main loop, matching p5.js)
          if (poissonMode === 'simultaneous') {
            let rhoOther = 0;
            for (let n = 0; n < nShells; n++) {
              if (n !== m) rhoOther += U[n][id] * U[n][id] * W[n][id];
            }
            const lapP = (P[m][ip] + P[m][im] + P[m][jp] + P[m][jm] + P[m][kp] + P[m][km] - 6 * P[m][id]) / h2;
            P[m][id] += dt * lapP + 2 * Math.PI * dt * rhoOther;
          }
        }
      }
    }
  }

  // Multigrid V-cycle Poisson solve (only if not simultaneous)
  if (poissonMode !== 'simultaneous') {
    for (let m = 0; m < nShells; m++) {
      const rhs = new Float32Array(S3);
      for (let i = 1; i < NN; i++)
        for (let j = 1; j < NN; j++)
          for (let k = 1; k < NN; k++) {
            const id = idx(i, j, k);
            let rhoOther = 0;
            for (let n = 0; n < nShells; n++) {
              if (n !== m) rhoOther += U[n][id] * U[n][id] * W[n][id];
            }
            rhs[id] = 2 * Math.PI * rhoOther;
          }
      poissonVcycle(P[m], rhs);
    }
  }

  // Normalize
  for (let m = 0; m < nShells; m++) {
    let n2 = 0;
    for (let i = 1; i < NN; i++)
      for (let j = 1; j < NN; j++)
        for (let k = 1; k < NN; k++)
          n2 += U[m][idx(i, j, k)] ** 2 * h3;
    const sc = 1 / Math.sqrt(n2);
    for (let i = 1; i < NN; i++)
      for (let j = 1; j < NN; j++)
        for (let k = 1; k < NN; k++)
          U[m][idx(i, j, k)] *= sc;
  }

  // Sample both densities at the inner shell boundary R = Rmax[0]
  let bdyInfo = '';
  const Rb = shellConfig[0].Rmax;
  const ib = Math.round(N2 + Rb / h);
  if (ib > 0 && ib < NN) {
    const id = idx(ib, N2, N2);
    bdyInfo = ' U0(' + Rb.toFixed(1) + ')=' + U[0][id].toFixed(4) + ' U1(' + Rb.toFixed(1) + ')=' + U[1][id].toFixed(4);
  }

  return { E: E_T + E_eK + E_ee, E_T, E_eK, E_ee, bdyInfo };
}

// Expose for visualization
window._atom = { U, W, P, K, NN, N2, S, S2, S3, h, nShells, shellConfig, physicsStep, idx };
