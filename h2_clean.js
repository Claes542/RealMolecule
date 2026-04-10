// h2_clean.js — H2 with w-weighted Laplacian, matching p5 exactly
// CPU solver, Gauss-Seidel (in-place), w-weighted energy
"use strict";

const NN = window.USER_NN || 100;
const screenAu = window.USER_SCREEN || 10;
const MAX_STEPS = window.USER_STEPS || 3000;

const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.floor(NN / 2);
const h = screenAu / NN;

// Full sweep
const SWEEP_R = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 6.0];
let sweepIdx = 0;
let sweepResults = [];
let D_CELLS = Math.round(SWEEP_R[0] / h);
const h2 = h * h;
const h3 = h * h * h;
const d = 0.1;
const dt = d * h2;
const half_d = 0.5 * d;

let nuc1_i, nuc2_i, R_au, V_KK;
let E_T = 0, E_eK = 0, E_ee = 0, E_tot = 0;
let stepCount = 0;

const K = new Float32Array(S3);
const u1 = new Float32Array(S3);
const u2 = new Float32Array(S3);
const w1 = new Float32Array(S3);
const w2 = new Float32Array(S3);
const P1 = new Float32Array(S3);
const P2 = new Float32Array(S3);

function initSim(D) {
  D_CELLS = D;
  nuc1_i = N2 - Math.round(D / 2);
  nuc2_i = N2 + Math.round(D / 2);
  R_au = D * h;
  V_KK = 1.0 / Math.sqrt(R_au * R_au + 0.12 * h2);
  K.fill(0); u1.fill(0); u2.fill(0); w1.fill(0); w2.fill(0); P1.fill(0); P2.fill(0);
  for (let i = 0; i <= NN; i++) {
    for (let j = 0; j <= NN; j++) {
      for (let k = 0; k <= NN; k++) {
        const id = i * S2 + j * S + k;
        const dx1 = (i - nuc1_i) * h, dy1 = (j - N2) * h, dz1 = (k - N2) * h;
        const dx2 = (i - nuc2_i) * h, dy2 = (j - N2) * h, dz2 = (k - N2) * h;
        const r1 = Math.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1 + h2);
        const r2 = Math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2 + h2);
        K[id] = 1.0/r1 + 1.0/r2;
        if (i < N2) { u1[id] = Math.exp(-r1); w1[id] = 1; P2[id] = 0.5/r1; }
        if (i > N2) { u2[id] = Math.exp(-r2); w2[id] = 1; P1[id] = 0.5/r2; }
      }
    }
  }
  stepCount = 0;
  console.log(`Init R=${R_au.toFixed(2)} au (D=${D}) V_KK=${V_KK.toFixed(4)}`);
}
initSim(Math.round(SWEEP_R[0] / h));

function doStep() {
  // All updates in-place (Gauss-Seidel) — matching p5 exactly
  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = i*S2 + j*S + k;

        // c = 0.5*(u_self - u_other)
        const c1 = 0.5 * (u1[id] - u2[id]);
        const c2 = 0.5 * (u2[id] - u1[id]);

        // Update w1
        const w1c = w1[id];
        const lap_w1 = w1[id+S2]+w1[id-S2]+w1[id+S]+w1[id-S]+w1[id+1]+w1[id-1] - 6*w1c;
        const gw1x = (w1[id+S2]-w1[id-S2])/h;
        const gw1y = (w1[id+S]-w1[id-S])/h;
        const gw1z = (w1[id+1]-w1[id-1])/h;
        const gradw1 = Math.sqrt(gw1x*gw1x + gw1y*gw1y + gw1z*gw1z);
        w1[id] = w1c + 2*dt*Math.abs(c1)*lap_w1/h2 + 10*dt*c1*gradw1;

        // Update w2
        const w2c = w2[id];
        const lap_w2 = w2[id+S2]+w2[id-S2]+w2[id+S]+w2[id-S]+w2[id+1]+w2[id-1] - 6*w2c;
        const gw2x = (w2[id+S2]-w2[id-S2])/h;
        const gw2y = (w2[id+S]-w2[id-S])/h;
        const gw2z = (w2[id+1]-w2[id-1])/h;
        const gradw2 = Math.sqrt(gw2x*gw2x + gw2y*gw2y + gw2z*gw2z);
        w2[id] = w2c + 2*dt*Math.abs(c2)*lap_w2/h2 + 10*dt*c2*gradw2;

        // Update u1: w-weighted Laplacian + potential
        {
          const uc = u1[id];
          const wc = w1[id];
          const fxp = (u1[id+S2]-uc)*(w1[id+S2]+wc)*0.5;
          const fxm = (uc-u1[id-S2])*(wc+w1[id-S2])*0.5;
          const fyp = (u1[id+S]-uc)*(w1[id+S]+wc)*0.5;
          const fym = (uc-u1[id-S])*(wc+w1[id-S])*0.5;
          const fzp = (u1[id+1]-uc)*(w1[id+1]+wc)*0.5;
          const fzm = (uc-u1[id-1])*(wc+w1[id-1])*0.5;
          const wlap = (fxp-fxm) + (fyp-fym) + (fzp-fzm);
          u1[id] = uc + half_d*wlap + dt*(K[id]-2*P1[id])*uc*wc;
        }

        // Update u2: w-weighted Laplacian + potential
        {
          const uc = u2[id];
          const wc = w2[id];
          const fxp = (u2[id+S2]-uc)*(w2[id+S2]+wc)*0.5;
          const fxm = (uc-u2[id-S2])*(wc+w2[id-S2])*0.5;
          const fyp = (u2[id+S]-uc)*(w2[id+S]+wc)*0.5;
          const fym = (uc-u2[id-S])*(wc+w2[id-S])*0.5;
          const fzp = (u2[id+1]-uc)*(w2[id+1]+wc)*0.5;
          const fzm = (uc-u2[id-1])*(wc+w2[id-1])*0.5;
          const wlap = (fxp-fxm) + (fyp-fym) + (fzp-fzm);
          u2[id] = uc + half_d*wlap + dt*(K[id]-2*P2[id])*uc*wc;
        }

        // Update P in-place (Gauss-Seidel)
        {
          const Pc = P1[id];
          const lapP = P1[id+S2]+P1[id-S2]+P1[id+S]+P1[id-S]+P1[id+1]+P1[id-1]-6*Pc;
          P1[id] = Pc + dt*(lapP/h2 + 2*Math.PI*u2[id]*u2[id]);
        }
        {
          const Pc = P2[id];
          const lapP = P2[id+S2]+P2[id-S2]+P2[id+S]+P2[id-S]+P2[id+1]+P2[id-1]-6*Pc;
          P2[id] = Pc + dt*(lapP/h2 + 2*Math.PI*u1[id]*u1[id]);
        }
      }
    }
  }

  // Normalize over all cells
  let norm1 = 0, norm2 = 0;
  for (let id = 0; id < S3; id++) {
    norm1 += u1[id]*u1[id]*h3;
    norm2 += u2[id]*u2[id]*h3;
  }
  if (norm1 > 0) { const s = 1/Math.sqrt(norm1); for (let id = 0; id < S3; id++) u1[id] *= s; }
  if (norm2 > 0) { const s = 1/Math.sqrt(norm2); for (let id = 0; id < S3; id++) u2[id] *= s; }

  stepCount++;
}

function computeEnergy() {
  E_T = 0; E_eK = 0; E_ee = 0;
  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = i*S2 + j*S + k;
        for (let m = 0; m < 2; m++) {
          const u = m === 0 ? u1 : u2;
          const w = m === 0 ? w1 : w2;
          const P = m === 0 ? P1 : P2;
          const v = u[id];
          const wv = w[id];
          // w-weighted energy
          const gx = u[id+S2] - v;
          const gy = u[id+S] - v;
          const gz = u[id+1] - v;
          E_T += 0.5 * wv * (gx*gx + gy*gy + gz*gz) * h;
          E_eK += -K[id] * wv * v * v * h3;
          E_ee += P[id] * wv * v * v * h3;
        }
      }
    }
  }
  E_tot = E_T + E_eK + E_ee + V_KK;
}

// p5.js draw
window.setup = function() { createCanvas(700, 500); };

window.draw = function() {
  background(220);
  if (stepCount < MAX_STEPS) {
    doStep();
    if (stepCount % 10 === 0) computeEnergy();
  } else if (sweepIdx < SWEEP_R.length) {
    // Record result and move to next R
    computeEnergy();
    sweepResults.push({ R: R_au, E: E_tot, T: E_T, VeK: E_eK, Vee: E_ee, VKK: V_KK });
    console.log(`SWEEP R=${R_au.toFixed(2)} E=${E_tot.toFixed(4)} T=${E_T.toFixed(4)} VeK=${E_eK.toFixed(4)} Vee=${E_ee.toFixed(4)}`);
    sweepIdx++;
    if (sweepIdx < SWEEP_R.length) {
      initSim(Math.round(SWEEP_R[sweepIdx] / h));
    } else {
      // Done — compute binding
      const Eref = sweepResults[sweepResults.length-1].E;
      const best = sweepResults.reduce((a,b) => a.E < b.E ? a : b);
      console.log("=== SWEEP DONE ===");
      for (const r of sweepResults)
        console.log(`  R=${r.R.toFixed(2)} E=${r.E.toFixed(4)} E_bind=${(r.E-Eref).toFixed(4)}`);
      console.log(`BEST: R=${best.R.toFixed(2)} E_bind=${(best.E-Eref).toFixed(4)} (exact: -0.1745)`);
    }
  }

  const plotW = 400, plotX = 10;
  noStroke();
  for (let i = 0; i < S; i++) {
    const id = i*S2 + N2*S + N2;
    const x = plotX + i*(plotW/S);
    // w
    fill(0); ellipse(x, 100-80*w1[id], 3); ellipse(x, 100-80*w2[id], 3);
    // u
    fill(255,0,0); ellipse(x, 250-150*u1[id], 3);
    fill(0,0,255); ellipse(x, 250-150*u2[id], 3);
    // P, K
    fill(255,0,0,150); ellipse(x, 420-80*P1[id], 2);
    fill(0,0,255,150); ellipse(x, 420-80*P2[id], 2);
    fill(0,180,0); ellipse(x, 420-30*K[id], 3);
  }

  // Boundary & nuclei
  stroke(150); line(plotX+N2*(plotW/S), 50, plotX+N2*(plotW/S), 450); noStroke();
  fill(0);
  circle(plotX+nuc1_i*(plotW/S), 250, 8);
  circle(plotX+nuc2_i*(plotW/S), 250, 8);

  // Labels
  fill(0); textSize(13);
  text("H2 w-weighted (Gauss-Seidel) N="+NN+" R="+R_au.toFixed(2)+" au", 10, 20);
  text("step="+stepCount, 10, 38);
  fill(0); text("w1,w2 (black)", plotX+5, 48);
  fill(255,0,0); text("u1", plotX+5, 148); fill(0,0,255); text("u2", plotX+25, 148);
  fill(0); text("P1,P2,K(green)", plotX+5, 340);

  textSize(12); fill(0);
  text("T = "+E_T.toFixed(4), 430, 140);
  text("V_eK = "+E_eK.toFixed(4), 430, 158);
  text("V_ee = "+E_ee.toFixed(4), 430, 176);
  text("V_KK = "+V_KK.toFixed(4), 430, 194);
  textSize(14);
  text("E = "+E_tot.toFixed(4), 430, 220);
  textSize(11);
  text("(exact: -1.1745)", 430, 240);
  text("K = 1/sqrt(r²+0.12h²)", 430, 270);
  text("w-weighted energy", 430, 290);

  // Sweep progress
  textSize(11); fill(0);
  text("Sweep: " + sweepIdx + "/" + SWEEP_R.length + "  R=" + R_au.toFixed(2) + "  step=" + stepCount + "/" + MAX_STEPS, 430, 310);
  if (sweepResults.length > 0) {
    const Eref = sweepResults[sweepResults.length-1].E;
    let ty = 330;
    text("R(au)  E(Ha)   E_bind", 430, ty); ty += 14;
    for (const r of sweepResults) {
      const bind = r.E - Eref;
      text(r.R.toFixed(1)+"   "+r.E.toFixed(4)+"  "+bind.toFixed(4), 430, ty); ty += 13;
    }
  }

  if (stepCount > 0 && stepCount % 200 === 0)
    console.log("step="+stepCount+" E="+E_tot.toFixed(6)+" T="+E_T.toFixed(4)+" VeK="+E_eK.toFixed(4)+" Vee="+E_ee.toFixed(4));
};
