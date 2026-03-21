"""
RealQM 3D solver — CuPy GPU implementation
Port of molecule.js WebGPU compute shaders to CUDA via CuPy.

Core algorithm: Imaginary-time propagation on 3D grid
  U_new = U + 0.5*dv*lap + dv*(K - 2P)*U
  with domain decomposition, Poisson solve, SIC, nuclear dynamics.
"""

import cupy as cp
import numpy as np
import json, time

# ─── CUDA Kernels ───

_init_trial_kernel = cp.RawKernel(r'''
extern "C" __global__
void initTrial(float* U, int* label, float* K_buf, float* bestU,
               const float* atoms, int nAtoms, int S, float h, float h2) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) {
        U[id] = 0.0f; label[id] = 0; K_buf[id] = 0.0f; bestU[id] = 0.0f;
        return;
    }
    float xi = i * h, yj = j * h, zk = k * h;
    float bU = 0.0f;
    int bestN = 0;
    float Kval = 0.0f;
    float inv_sqrt_pi = 0.5641895835f;
    for (int n = 0; n < nAtoms; n++) {
        float Za = atoms[n * 5 + 3];
        if (Za <= 0.0f) continue;
        float rc = atoms[n * 5 + 4];
        float dx = xi - atoms[n * 5 + 0] * h;
        float dy = yj - atoms[n * 5 + 1] * h;
        float dz = zk - atoms[n * 5 + 2] * h;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r;
        if (rc > 0.0f) {
            r = fmaxf(sqrtf(r2 + 0.04f * h2), rc);
        } else {
            r = sqrtf(r2 + h2);
        }
        Kval += Za / r;
        float uTrial = Za * Za * inv_sqrt_pi * expf(-Za * r);
        if (uTrial > bU) { bU = uTrial; bestN = n; }
    }
    K_buf[id] = Kval;
    bestU[id] = bU;
    label[id] = bestN;
    U[id] = bU;
}
''', 'initTrial')

_update_kernel = cp.RawKernel(r'''
extern "C" __global__
void updateU(const float* U, float* Unew, const float* K, const float* P,
             const int* label, int S, float half_dv, float dtv) {
    // half_dv = 0.5*dv, dtv = dv*h^2
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) {
        Unew[id] = 0.0f;
        return;
    }
    int myLabel = label[id];
    float u = U[id];
    // Laplacian with Neumann BC at domain boundaries (not divided by h^2)
    float uip = (label[id + S*S] == myLabel) ? U[id + S*S] : u;
    float uim = (label[id - S*S] == myLabel) ? U[id - S*S] : u;
    float ujp = (label[id + S] == myLabel) ? U[id + S] : u;
    float ujm = (label[id - S] == myLabel) ? U[id - S] : u;
    float ukp = (label[id + 1] == myLabel) ? U[id + 1] : u;
    float ukm = (label[id - 1] == myLabel) ? U[id - 1] : u;
    float lap = uip + uim + ujp + ujm + ukp + ukm - 6.0f * u;
    // ITP: U_new = U + 0.5*dv*lap + dv*h^2*(K - 2P)*U
    Unew[id] = u + half_dv * lap + dtv * (K[id] - 2.0f * P[id]) * u;
}
''', 'updateU')

_boundary_evolve_kernel = cp.RawKernel(r'''
extern "C" __global__
void boundaryEvolve(const float* U, int* label, const float* atoms,
                    int nAtoms, int S, float h, float h2, float h3,
                    float boundarySpeed) {
    // For each grid point, compute density-weighted score for each domain
    // and flip label if another domain has higher score
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) return;

    float xi = i * h, yj = j * h, zk = k * h;
    float u2 = U[id] * U[id];
    int curLabel = label[id];

    // Count neighbors in each domain
    int myCount = 0, otherBest = -1;
    int otherCount = 0;
    int neighbors[6] = {id+S*S, id-S*S, id+S, id-S, id+1, id-1};
    for (int n = 0; n < 6; n++) {
        if (label[neighbors[n]] == curLabel) myCount++;
        else {
            otherCount++;
            otherBest = label[neighbors[n]];
        }
    }

    // Only consider flipping at domain boundaries (has at least 1 other-domain neighbor)
    if (otherCount == 0) return;

    // Compare density-weighted nuclear attraction: Z/r * u^2
    float curScore = 0.0f, bestScore = 0.0f;
    int bestN = curLabel;
    for (int n = 0; n < nAtoms; n++) {
        float Za = atoms[n * 5 + 3];
        if (Za <= 0.0f) continue;
        float dx = xi - atoms[n * 5 + 0] * h;
        float dy = yj - atoms[n * 5 + 1] * h;
        float dz = zk - atoms[n * 5 + 2] * h;
        float r2 = dx*dx + dy*dy + dz*dz + h2;
        float score = Za / sqrtf(r2);
        if (n == curLabel) curScore = score;
        if (score > bestScore) { bestScore = score; bestN = n; }
    }

    // Flip if another nucleus has stronger attraction and enough neighbors agree
    if (bestN != curLabel && bestScore > curScore * 1.1f && otherCount >= 2) {
        label[id] = bestN;
    }
}
''', 'boundaryEvolve')

_recompute_K_kernel = cp.RawKernel(r'''
extern "C" __global__
void recomputeK(float* K, const float* atoms, int nAtoms, int S, float h, float h2) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) {
        K[id] = 0.0f; return;
    }
    float xi = i * h, yj = j * h, zk = k * h;
    float Kval = 0.0f;
    for (int n = 0; n < nAtoms; n++) {
        float Za = atoms[n * 5 + 3];
        if (Za <= 0.0f) continue;
        float rc = atoms[n * 5 + 4];
        float dx = xi - atoms[n * 5 + 0] * h;
        float dy = yj - atoms[n * 5 + 1] * h;
        float dz = zk - atoms[n * 5 + 2] * h;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r;
        if (rc > 0.0f) r = fmaxf(sqrtf(r2 + 0.04f * h2), rc);
        else r = sqrtf(r2 + h2);
        Kval += Za / r;
    }
    K[id] = Kval;
}
''', 'recomputeK')

_compute_rho_kernel = cp.RawKernel(r'''
extern "C" __global__
void computeRho(float* rho, const float* U, int S3) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= S3) return;
    rho[id] = U[id] * U[id];
}
''', 'computeRho')

_jacobi_kernel = cp.RawKernel(r'''
extern "C" __global__
void jacobiStep(const float* P, float* Pnew, const float* rho,
                int S, float h2, float coeff) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) {
        Pnew[id] = 0.0f;
        return;
    }
    float sumN = P[id+S*S] + P[id-S*S] + P[id+S] + P[id-S] + P[id+1] + P[id-1];
    Pnew[id] = (sumN + coeff * rho[id] * h2) / 6.0f;
}
''', 'jacobiStep')

_normalize_kernel = cp.RawKernel(r'''
extern "C" __global__
void normalizeU(float* U, const int* label, const float* norms, const float* Zeff,
                int S3, int nAtoms, float h3) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= S3) return;
    int lbl = label[id];
    if (lbl < 0 || lbl >= nAtoms) return;
    float z = Zeff[lbl];
    if (z <= 0.0f) return;
    float n = norms[lbl];
    if (n > 1e-20f) {
        U[id] *= sqrtf(z / n);
    }
}
''', 'normalizeU')

_energy_kernel = cp.RawKernel(r'''
extern "C" __global__
void computeEnergy(const float* U, const float* K, const float* P,
                   const int* label, float* out, int S, float h, float h3) {
    // out[0] = T, out[1] = V_eK, out[2] = V_ee
    // Use shared memory for block reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + tid;
    int S3 = S * S * S;
    int NN = S - 1;

    float localT = 0.0f, localVeK = 0.0f, localVee = 0.0f;

    if (id < S3) {
        int i = id / (S * S);
        int j = (id / S) % S;
        int k = id % S;
        if (i >= 1 && i < NN && j >= 1 && j < NN && k >= 1 && k < NN) {
            float u = U[id];
            float rho = u * u;
            int myLabel = label[id];
            // Kinetic: 0.5 * |grad U|^2 * h (forward differences)
            float dui = (label[id+S*S]==myLabel) ? U[id+S*S] - u : 0.0f;
            float duj = (label[id+S]==myLabel) ? U[id+S] - u : 0.0f;
            float duk = (label[id+1]==myLabel) ? U[id+1] - u : 0.0f;
            localT = 0.5f * (dui*dui + duj*duj + duk*duk) * h;
            // Potential energies
            localVeK = -K[id] * rho * h3;
            localVee = P[id] * rho * h3;
        }
    }

    sdata[tid] = localT;
    sdata[tid + blockDim.x] = localVeK;
    sdata[tid + 2*blockDim.x] = localVee;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
            sdata[tid + 2*blockDim.x] += sdata[tid + 2*blockDim.x + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[0], sdata[0]);
        atomicAdd(&out[1], sdata[blockDim.x]);
        atomicAdd(&out[2], sdata[2*blockDim.x]);
    }
}
''', 'computeEnergy')

_force_kernel_batched = cp.RawKernel(r'''
extern "C" __global__
void computeForceAll(const float* U, float* forces, const float* atoms,
                     int nAtoms, int S, float h, float h2, float h3) {
    // forces[n*3+0..2] = Fx,Fy,Fz for atom n
    // Each thread processes one grid point and contributes to ALL atoms
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int S3 = S * S * S;
    if (id >= S3) return;
    int NN = S - 1;
    int i = id / (S * S);
    int j = (id / S) % S;
    int k = id % S;
    if (i < 1 || i >= NN || j < 1 || j >= NN || k < 1 || k >= NN) return;

    float rho = U[id] * U[id];
    if (rho < 1e-20f) return;
    float xi = i * h, yj = j * h, zk = k * h;

    for (int n = 0; n < nAtoms; n++) {
        float Za = atoms[n * 5 + 3];
        if (Za <= 0.0f) continue;
        float dx = xi - atoms[n * 5 + 0] * h;
        float dy = yj - atoms[n * 5 + 1] * h;
        float dz = zk - atoms[n * 5 + 2] * h;
        float r2 = dx*dx + dy*dy + dz*dz + h2;
        float inv_r3 = 1.0f / (r2 * sqrtf(r2));
        float w = Za * rho * inv_r3 * h3;
        atomicAdd(&forces[n * 3 + 0], w * dx);
        atomicAdd(&forces[n * 3 + 1], w * dy);
        atomicAdd(&forces[n * 3 + 2], w * dz);
    }
}
''', 'computeForceAll')


class RealQMSolver:
    """GPU-accelerated 3D quantum solver using CuPy."""

    def __init__(self, config):
        self.NN = config['gridN']
        self.S = self.NN + 1
        self.S3 = self.S ** 3
        self.screen = config['screen']
        self.h = self.screen / self.NN
        self.h2 = self.h * self.h
        self.h3 = self.h ** 3
        self.N2 = self.NN // 2

        # Atoms: list of {i, j, k, Z, rc, el}
        atoms = config['atoms']
        self.nAtoms = len(atoms)
        self.Z_eff = np.array([a['Z'] for a in atoms], dtype=np.float32)
        self.atom_pos = np.array([[a['i'], a['j'], a.get('k', self.N2)] for a in atoms], dtype=np.float32)
        self.atom_rc = np.array([a.get('rc', 0) for a in atoms], dtype=np.float32)
        self.atom_el = [a.get('el', '') for a in atoms]
        self.active = self.Z_eff > 0
        self.nElec = int(self.active.sum())

        # Timestep — conservative for stability
        if self.nElec > 500:
            self.dv = 0.005
        elif self.nElec > 200:
            self.dv = 0.01
        elif self.nElec > 100:
            self.dv = 0.02
        elif self.nElec > 30:
            self.dv = 0.03
        else:
            self.dv = 0.12
        self.half_dv = 0.5 * self.dv
        self.dtv = self.dv * self.h2  # dt = dv * h^2 for potential term

        # Dynamics params
        self.force_scale = config.get('forceScale', 1.0)
        self.damping = config.get('damping', 0.98)
        self.dt_nuc = config.get('dt_nuc', 0.8 if self.nElec <= 5 else 0.2 if self.nElec > 200 else 0.8)
        self.max_vel = 0.3 if self.nElec <= 5 else 0.03 if self.nElec > 200 else 0.1
        self.dynamics_enabled = config.get('dynamics', False)
        self._nProtein = config.get('nProtein', self.nAtoms)
        self._config_force_interval = config.get('force_interval', 500)

        # GPU arrays
        self.U = cp.zeros(self.S3, dtype=cp.float32)
        self.U2 = cp.zeros(self.S3, dtype=cp.float32)
        self.K_buf = cp.zeros(self.S3, dtype=cp.float32)
        self.P = cp.zeros(self.S3, dtype=cp.float32)
        self.P2 = cp.zeros(self.S3, dtype=cp.float32)
        self.rho = cp.zeros(self.S3, dtype=cp.float32)
        self.label = cp.zeros(self.S3, dtype=cp.int32)
        self.bestU = cp.zeros(self.S3, dtype=cp.float32)

        # Nuclear dynamics arrays
        self.nuc_vel = np.zeros((self.nAtoms, 3), dtype=np.float32)
        self.nuc_force = np.zeros((self.nAtoms, 3), dtype=np.float32)

        # Build atom buffer for GPU
        atom_data = np.zeros((self.nAtoms, 5), dtype=np.float32)
        for n, a in enumerate(atoms):
            atom_data[n] = [a['i'], a['j'], a.get('k', self.N2), a['Z'], a.get('rc', 0)]
        self.atom_data_gpu = cp.asarray(atom_data.flatten())

        # Thread config
        self.block = 256
        self.grid = (self.S3 + self.block - 1) // self.block

        # Energy
        self.E_T = 0.0
        self.E_eK = 0.0
        self.E_ee = 0.0
        self.E_KK = 0.0
        self.E = 0.0

        # Isolated atom energies for binding energy
        self._atom_ref = {'H': -0.5, 'O': -2.04, 'C': -5.43, 'N': -3.64}
        self.E_atoms_sum = sum(self._atom_ref.get(a.get('el', ''), 0) for a in atoms if a['Z'] > 0)

        self._Z_gpu = cp.asarray(self.Z_eff)
        self._initialized = False

    def initialize(self):
        """Initialize wavefunctions and nuclear potential on GPU."""
        _init_trial_kernel(
            (self.grid,), (self.block,),
            (self.U, self.label, self.K_buf, self.bestU,
             self.atom_data_gpu, np.int32(self.nAtoms),
             np.int32(self.S), np.float32(self.h), np.float32(self.h2))
        )
        cp.cuda.Stream.null.synchronize()

        # Normalize initial wavefunctions
        self._normalize()

        # Bootstrap Poisson solve so P isn't zero on first ITP step
        for _ in range(100):
            self._poisson_step()

        self._initialized = True

    def _normalize(self):
        """Normalize U per domain so ∫U²dV = Z_eff."""
        # Compute per-domain norms entirely on GPU using scatter_add
        rho = self.U * self.U * self.h3
        norms = cp.zeros(self.nAtoms, dtype=cp.float32)
        import cupyx
        cupyx.scatter_add(norms, self.label.clip(0, self.nAtoms - 1), rho)
        _normalize_kernel(
            (self.grid,), (self.block,),
            (self.U, self.label, norms, self._Z_gpu,
             np.int32(self.S3), np.int32(self.nAtoms), np.float32(self.h3))
        )

    def _poisson_step(self):
        """Jacobi iteration for Poisson equation: ∇²P = -2π·ρ."""
        _compute_rho_kernel((self.grid,), (self.block,), (self.rho, self.U, np.int32(self.S3)))
        coeff = 2.0 * np.pi
        for _ in range(2):
            _jacobi_kernel(
                (self.grid,), (self.block,),
                (self.P, self.P2, self.rho,
                 np.int32(self.S), np.float32(self.h2), np.float32(coeff))
            )
            self.P, self.P2 = self.P2, self.P

    def _compute_energy(self):
        """Compute kinetic, electron-nuclear, and electron-electron energies."""
        out = cp.zeros(3, dtype=cp.float32)
        smem = 3 * self.block * 4  # 3 floats per thread
        _energy_kernel(
            (self.grid,), (self.block,),
            (self.U, self.K_buf, self.P, self.label, out,
             np.int32(self.S), np.float32(self.h), np.float32(self.h3)),
            shared_mem=smem
        )
        cp.cuda.Stream.null.synchronize()
        vals = out.get()
        self.E_T = float(vals[0])
        self.E_eK = float(vals[1])
        self.E_ee = float(vals[2])

        # Nuclear-nuclear repulsion (CPU, small)
        self.E_KK = 0.0
        for a in range(self.nAtoms):
            if self.Z_eff[a] <= 0:
                continue
            for b in range(a + 1, self.nAtoms):
                if self.Z_eff[b] <= 0:
                    continue
                dx = (self.atom_pos[a][0] - self.atom_pos[b][0]) * self.h
                dy = (self.atom_pos[a][1] - self.atom_pos[b][1]) * self.h
                dz = (self.atom_pos[a][2] - self.atom_pos[b][2]) * self.h
                r = np.sqrt(dx*dx + dy*dy + dz*dz + self.h2)
                self.E_KK += self.Z_eff[a] * self.Z_eff[b] / r

        self.E = self.E_T + self.E_eK + self.E_ee + self.E_KK

    def _compute_forces(self):
        """Hellmann-Feynman forces — batched, protein atoms only."""
        BATCH = 32
        nForce = self._nProtein if hasattr(self, '_nProtein') else self.nAtoms
        self.nuc_force = np.zeros((self.nAtoms, 3), dtype=np.float32)
        for b0 in range(0, nForce, BATCH):
            b1 = min(b0 + BATCH, self.nAtoms)
            batchSize = b1 - b0
            # Build sub-atom array for this batch
            batch_data = np.zeros(batchSize * 5, dtype=np.float32)
            for n in range(batchSize):
                idx = b0 + n
                batch_data[n*5:n*5+5] = [self.atom_pos[idx][0], self.atom_pos[idx][1],
                                          self.atom_pos[idx][2], self.Z_eff[idx], self.atom_rc[idx]]
            batch_gpu = cp.asarray(batch_data)
            forces_gpu = cp.zeros(batchSize * 3, dtype=cp.float32)
            _force_kernel_batched(
                (self.grid,), (self.block,),
                (self.U, forces_gpu, batch_gpu,
                 np.int32(batchSize), np.int32(self.S),
                 np.float32(self.h), np.float32(self.h2), np.float32(self.h3))
            )
            cp.cuda.Stream.null.synchronize()
            f = forces_gpu.get().reshape(batchSize, 3)
            self.nuc_force[b0:b1] = f

        # Add nuclear-nuclear forces (CPU, small)
        nForce2 = self._nProtein if hasattr(self, '_nProtein') else self.nAtoms
        for n in range(nForce2):
            if self.Z_eff[n] <= 0:
                continue
            for m in range(self.nAtoms):
                if m == n or self.Z_eff[m] <= 0:
                    continue
                dx = (self.atom_pos[n][0] - self.atom_pos[m][0]) * self.h
                dy = (self.atom_pos[n][1] - self.atom_pos[m][1]) * self.h
                dz = (self.atom_pos[n][2] - self.atom_pos[m][2]) * self.h
                r2 = dx*dx + dy*dy + dz*dz + self.h2
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                ff = self.Z_eff[n] * self.Z_eff[m] * inv_r3
                self.nuc_force[n][0] += ff * dx
                self.nuc_force[n][1] += ff * dy
                self.nuc_force[n][2] += ff * dz

    def _move_nuclei(self):
        """Velocity Verlet nuclear dynamics — protein atoms only."""
        mass_map = {1: 1*1836, 2: 16*1836, 3: 14*1836, 4: 12*1836}
        nMove = self._nProtein if hasattr(self, '_nProtein') else self.nAtoms
        for n in range(nMove):
            z = int(self.Z_eff[n])
            if z <= 0:
                continue
            m = mass_map.get(z, 1836)
            for d in range(3):
                self.nuc_vel[n][d] += self.nuc_force[n][d] / m * self.dt_nuc * self.force_scale
                self.nuc_vel[n][d] *= self.damping
                self.nuc_vel[n][d] = np.clip(self.nuc_vel[n][d], -self.max_vel, self.max_vel)
                self.atom_pos[n][d] += self.nuc_vel[n][d] * self.dt_nuc / self.h
                self.atom_pos[n][d] = np.clip(self.atom_pos[n][d], 5, self.NN - 5)

        # Rebuild K buffer and atom data after nuclear motion
        atom_data = np.zeros(self.nAtoms * 5, dtype=np.float32)
        for n in range(self.nAtoms):
            atom_data[n*5:n*5+5] = [self.atom_pos[n][0], self.atom_pos[n][1],
                                     self.atom_pos[n][2], self.Z_eff[n], self.atom_rc[n]]
        self.atom_data_gpu = cp.asarray(atom_data)
        _recompute_K_kernel(
            (self.grid,), (self.block,),
            (self.K_buf, self.atom_data_gpu, np.int32(self.nAtoms),
             np.int32(self.S), np.float32(self.h), np.float32(self.h2))
        )

    def run(self, total_steps, norm_interval=20, poisson_interval=2,
            force_interval=None, report_interval=500):
        """Run the solver for total_steps. Returns list of snapshots."""
        if force_interval is None:
            force_interval = self._config_force_interval
        if not self._initialized:
            self.initialize()

        results = []
        t0 = time.time()

        for step in range(1, total_steps + 1):
            # Poisson solve
            if step % poisson_interval == 0:
                self._poisson_step()

            # Wavefunction update (ITP)
            _update_kernel(
                (self.grid,), (self.block,),
                (self.U, self.U2, self.K_buf, self.P, self.label,
                 np.int32(self.S), np.float32(self.half_dv), np.float32(self.dtv))
            )
            self.U, self.U2 = self.U2, self.U

            # Normalize
            if step % norm_interval == 0:
                self._normalize()

            # Boundary evolution — reassign domain labels (less frequent)
            if step % (norm_interval * 5) == 0:
                _boundary_evolve_kernel(
                    (self.grid,), (self.block,),
                    (self.U, self.label, self.atom_data_gpu,
                     np.int32(self.nAtoms), np.int32(self.S),
                     np.float32(self.h), np.float32(self.h2), np.float32(self.h3),
                     np.float32(0.5))
                )

            # Forces and dynamics
            if self.dynamics_enabled and step % force_interval == 0:
                self._compute_forces()
                self._move_nuclei()

            # Report
            if step % report_interval == 0:
                self._compute_energy()
                elapsed = time.time() - t0
                ms_per_step = elapsed / step * 1000

                snapshot = {
                    'step': int(step),
                    'E': float(self.E),
                    'E_T': float(self.E_T),
                    'E_eK': float(self.E_eK),
                    'E_ee': float(self.E_ee),
                    'E_KK': float(self.E_KK),
                    'E_bind': float(self.E - self.E_atoms_sum),
                    'ms_per_step': float(ms_per_step),
                    'elapsed_sec': float(elapsed),
                    'nucPos': [[float(x) for x in row] for row in self.atom_pos],
                }

                if hasattr(self, '_fold_atoms') and self._fold_atoms:
                    fa = self._fold_atoms
                    a = self.atom_pos[fa[0]] * self.h
                    b = self.atom_pos[fa[1]] * self.h
                    c = self.atom_pos[fa[2]] * self.h
                    ba = a - b
                    bc = c - b
                    dot = np.dot(ba, bc)
                    mag = np.linalg.norm(ba) * np.linalg.norm(bc)
                    if mag > 0:
                        angle = np.degrees(np.arccos(np.clip(dot / mag, -1, 1)))
                    else:
                        angle = 0
                    snapshot['fold_angle'] = float(angle)

                results.append(snapshot)
                fmax = float(np.max(np.linalg.norm(self.nuc_force[:self._nProtein], axis=1)))
                print(f"  Step {step}/{total_steps}: E={self.E:.6f} "
                      f"E_bind={self.E - self.E_atoms_sum:.6f} "
                      f"F_max={fmax:.4f} ({ms_per_step:.1f} ms/step)")

        cp.cuda.Stream.null.synchronize()
        return results

    def set_fold_atoms(self, indices):
        """Set 3 atom indices for fold angle measurement."""
        self._fold_atoms = indices
