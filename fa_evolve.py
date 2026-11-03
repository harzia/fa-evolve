"""
Evolving-basis Functional Attention dynamics (prototype v2).
 
Model. The field is u(x,t) = Phi(x; s(t)) c(t): G moving bumps (centres, widths in the basis state s) with
coefficients c. The integrated state is (s, z) with z = L^{-1} d in Cholesky coordinates, where d = Phi^T W u are
FA's dual (inner-product) coordinates and M = Phi^T W Phi = L L^T is the EXACT Gram matrix (no regulariser), so
the field energy is exactly 1/2 ||z||^2:
 
    z_dot = (K + N) z + g          N = L^{-1} H^T L^{-T} - L^{-1} L_dot   basis motion, exact geometry, skew
                                   K = L^{-1} C L                          physics as a linear generator
                                   g                                       physics as a forcing (generic heads)
    C_FA = Q~ K~^T (K~ K~^T + lam M)^{-1},   Q~ = Phi^T W F_Q(u),  K~ = Phi^T W F_K(u)
    s_dot = f_s(s, local field)    learned, speed-bounded; widths through tanh, valid for any state
 
Decoded velocity u_dot = (I - P) Phi_dot c + Phi M^{-1} (physics): motion outside the span + physics inside it.
 
Variants (--model). All share encoder, feature network (u, u_x, u_xx, u_xxx), basis, integrator and safeguards:
    full          moving basis + FA solve + transport                         (proposed)
    fixed         basis placed at t=0, then frozen                            does evolving the subspace help?
    no_transport  moving + FA solve, geometric term removed                    does the exact transport help?
    nosolve       moving + C = Q~ K~^T / (mean eigenvalue of K~K~^T + lam M)  does the least-squares SOLVE help?
                  (identical to full except the matrix inverse is replaced by a scalar: same linear form,
                   same implicit integration, same safeguards)
    field_head    moving + generic LOCAL head Phi^T W F(u, u_x, u_xx, u_xxx)  vs a learned local PDE right-hand side
    token_head    moving + generic NONLOCAL head: attention over basis tokens  vs a generic global-coupling network
    --physics_explicit  integrate FA / nosolve physics explicitly, exactly as the generic heads must be.
Safeguards are matched: --rate_cap bounds energy growth at the same rate for every head (||K||_F <= kappa for
linear heads, ||g|| <= kappa ||z|| for forcing heads); --stab makes energy non-increasing for every head.
 
Integration (--integrator cayley, default): explicit midpoint for s, implicit midpoint (Cayley) for z with the
generator frozen at the midpoint. With a linear generator, ||z_{n+1}||^2 - ||z_n||^2 = 2h zbar^T sym(K) zbar for
any step h: crossing bumps (nearly singular M, huge N) cannot inject energy.
 
All geometry (basis values, Gram matrices, projections, solves) is computed in float64; networks run in float32.
Normalisation constants and the speed bound use only the training snapshots (t = 0 and the seen times).
 
Usage:
    python fa_evolve.py --test
    python fa_evolve.py --problem kdv --model full --data data --out runs/kdv_full_G8_s0
"""
import argparse
import json
import math
import os
import time
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
TWO_PI = 2 * math.pi
PROBLEM_DEFAULTS = {
    "advect":  dict(basis="bump", dt=0.025, train_frac=0.5, w_min_frac=0.01),
    "kdv":     dict(basis="bump", dt=0.1,   train_frac=0.5, w_min_frac=0.02),
    "burgers": dict(basis="pou",  dt=0.025, train_frac=0.5, w_min_frac=0.01),
}
MODELS = ("full", "fixed", "no_transport", "nosolve", "field_head", "token_head")
F64 = True        # geometry in float64
 
 
def hi(t):
    return t.double() if (F64 and t.dtype != torch.float64) else t
 
 
# ============================================================================= data
class Trajectories:
    def __init__(self, path, stride, device):
        d = np.load(path)
        self.u = torch.tensor(d["u"][:, :, ::stride], dtype=torch.float32, device=device)   # (N, T, n)
        self.t = torch.tensor(d["t"], dtype=torch.float32)
        self.x = torch.tensor(d["x"][::stride], dtype=torch.float32, device=device)
        self.L = float(d["L"])
 
 
class Grid:
    """Uniform periodic quadrature grid: weights W = L/n, spectral derivatives up to third order."""
 
    def __init__(self, x, L):
        self.x, self.L, self.n = x, L, x.numel()
        self.w = L / self.n
        self.k = TWO_PI / L * torch.fft.rfftfreq(self.n, d=1.0 / self.n, dtype=x.dtype, device=x.device)
 
    def derivs(self, u):
        uh = torch.fft.rfft(u, dim=-1)
        ik = 1j * self.k
        return tuple(torch.fft.irfft(ik ** p * uh, n=self.n, dim=-1) for p in (1, 2, 3))
 
 
def time_splits(t, T_train):
    seen = [i for i in range(len(t)) if i > 0 and i % 2 == 0 and t[i] <= T_train + 1e-9]
    unseen = [i for i in range(len(t)) if i % 2 == 1 and t[i] <= T_train + 1e-9]
    extrap = [i for i in range(len(t)) if t[i] > T_train + 1e-9]
    return seen, unseen, extrap
 
 
def normalisation_scales(u, idx, grid, n_traj=64):
    """Std of u, u_x, u_xx, u_xxx using ONLY the allowed snapshot indices (no temporal leakage)."""
    with torch.no_grad():
        sample = u[:n_traj][:, idx].reshape(-1, grid.n)
        ders = grid.derivs(sample)
        return tuple(float(v.std()) + 1e-6 for v in (sample,) + ders)
 
 
def estimate_max_speed(u, t, L, idx, T_train, n_traj=64, n_win=8):
    """Structure speed from windowed cross-correlation of snapshot pairs drawn ONLY from the allowed indices."""
    u, t = u[:n_traj].cpu().numpy().astype(np.float64), t.cpu().numpy()
    idx = sorted(idx)
    n = u.shape[-1]
    dx, x = L / n, np.arange(n) * L / n
    step = t[idx[1]] - t[idx[0]]
    lag = max(1, int(round((T_train / 6) / step)))
    speeds = []
    for j in range(len(idx) - lag):
        ka, kb = idx[j], idx[j + lag]
        a, b = u[:, ka] - u[:, ka].mean(-1, keepdims=True), u[:, kb] - u[:, kb].mean(-1, keepdims=True)
        for c in np.arange(n_win) * L / n_win:
            win = np.exp(-(((x - c + L / 2) % L - L / 2) ** 2) / (2 * (L / n_win) ** 2))
            aw, bw = a * win, b * win
            energy = (aw ** 2).sum(-1)
            s = np.argmax(np.fft.irfft(np.fft.rfft(bw) * np.conj(np.fft.rfft(aw)), n=n), -1)
            s = np.where(s > n // 2, s - n, s)
            speeds.append((np.abs(s) * dx / (t[kb] - t[ka]))[energy > 0.05 * energy.max()])
    return float(np.percentile(np.concatenate(speeds), 95))
 
 
# ============================================================================= basis
class VonMisesBasis(nn.Module):
    """G smooth periodic bumps  exp(kappa (cos(2 pi (x - mu) / L) - 1)),  kappa = (L / (2 pi w))^2:
    a Gaussian of std w near its centre, and smooth and periodic everywhere (unlike a wrapped Gaussian, which has a
    kink at the antipode). Per bump: angle theta (mu = theta L / 2 pi) and eta = tanh(sigma) setting w in
    [w_min, w_max]. kind 'bump': as is; 'pou': softmax over bumps (a smooth partition of unity)."""
 
    def __init__(self, G, L, w_min, w_max, kind):
        super().__init__()
        self.G, self.L, self.w_min, self.w_max, self.kind = G, L, w_min, w_max, kind
 
    def width(self, eta):
        return self.w_min + (self.w_max - self.w_min) * (eta + 1) / 2
 
    def centre(self, theta):
        return torch.remainder(theta * self.L / TWO_PI, self.L)
 
    def forward(self, x, theta, eta, theta_dot=None, eta_dot=None):
        w = self.width(eta)
        kap = (self.L / (TWO_PI * w)) ** 2                                              # (B, G)
        xx = x.unsqueeze(0) if x.dim() == 1 else x
        ang = TWO_PI / self.L * (xx[:, :, None] - (theta * self.L / TWO_PI)[:, None, :])  # (B, n, G)
        cs = torch.cos(ang)
        ell = kap[:, None, :] * (cs - 1)
        Phi = torch.exp(ell) if self.kind == "bump" else torch.softmax(ell, dim=-1)
        if theta_dot is None:
            return Phi
        mu_dot = theta_dot * self.L / TWO_PI
        w_dot = (self.w_max - self.w_min) / 2 * eta_dot
        kap_dot = -2 * kap * w_dot / w
        ell_dot = kap_dot[:, None, :] * (cs - 1) + \
            kap[:, None, :] * torch.sin(ang) * (TWO_PI / self.L) * mu_dot[:, None, :]
        if self.kind == "bump":
            Phi_dot = Phi * ell_dot
        else:
            Phi_dot = Phi * (ell_dot - (Phi * ell_dot).sum(-1, keepdim=True))
        return Phi, Phi_dot
 
    def dx(self, x, theta, eta):
        """Analytic x-derivative of each basis function (used to test smoothness)."""
        w = self.width(eta)
        kap = (self.L / (TWO_PI * w)) ** 2
        xx = x.unsqueeze(0) if x.dim() == 1 else x
        ang = TWO_PI / self.L * (xx[:, :, None] - (theta * self.L / TWO_PI)[:, None, :])
        Phi = self.forward(x, theta, eta)
        ell_x = -kap[:, None, :] * torch.sin(ang) * TWO_PI / self.L
        if self.kind == "bump":
            return Phi * ell_x
        return Phi * (ell_x - (Phi * ell_x).sum(-1, keepdim=True))
 
 
def mlp(sizes, act=nn.GELU, last_zero=False):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    layers = layers[:-1]
    if last_zero:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)
 
 
def interp_periodic(u, pos, grid):
    idx = torch.remainder(pos, grid.L) / grid.L * grid.n
    i0 = torch.floor(idx).long() % grid.n
    frac = idx - torch.floor(idx)
    return (1 - frac) * torch.gather(u, 1, i0) + frac * torch.gather(u, 1, (i0 + 1) % grid.n)
 
 
# ============================================================================= small dense linear algebra (float64)
def solve(A, B):
    X, info = torch.linalg.solve_ex(A, B)
    if (info != 0).any():
        scale = A.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1e-300)
        X = torch.linalg.solve_ex(A + 1e-12 * scale[..., None, None] *
                                  torch.eye(A.shape[-1], dtype=A.dtype, device=A.device), B)[0]
    return X
 
 
def chol(M, counter=None):
    """Cholesky of the exact Gram matrix. A failure means the basis is numerically degenerate: retry with a tiny
    jitter so the step completes, and COUNT it (logged), since the exact-geometry guarantees do not hold there."""
    Lc, info = torch.linalg.cholesky_ex(M)
    bad = info != 0
    if bad.any():
        if counter is not None:
            counter[0] += int(bad.sum())
        scale = M.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-300)
        Lc = torch.linalg.cholesky_ex(M + 1e-12 * scale[..., None, None] *
                                      torch.eye(M.shape[-1], dtype=M.dtype, device=M.device))[0]
    return Lc
 
 
def to_chol(C, Lc):           # K = L^{-1} C L
    return torch.linalg.solve_triangular(Lc, C @ Lc, upper=False)
 
 
def from_chol(K, Lc):         # C = L K L^{-1}
    return Lc @ torch.linalg.solve_triangular(Lc.transpose(1, 2), K.transpose(1, 2), upper=True).transpose(1, 2)
 
 
def fa_correspondence(Qt, Kt, M, lam):
    """C = Q~ K~^T (K~ K~^T + lam M)^{-1}. The lam*M ridge transforms like K~K~^T, so C is frame covariant."""
    S = Kt @ Kt.transpose(1, 2) + lam * M
    return solve(S, (Qt @ Kt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
 
 
def nosolve_correspondence(Qt, Kt, M, lam):
    """Ablation of the solve: the matrix inverse replaced by the inverse of its mean eigenvalue (a scalar)."""
    S = Kt @ Kt.transpose(1, 2) + lam * M
    s = S.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-300)
    return Qt @ Kt.transpose(1, 2) / s[:, None, None]
 
 
def guard_generator(K, stab, cap):
    """Safeguards on a linear generator in Cholesky coordinates: --stab removes the positive part of sym(K)
    (energy non-increasing); --rate_cap scales so ||K||_F <= cap (energy growth <= exp(2 cap t))."""
    if stab:
        S = 0.5 * (K + K.transpose(1, 2))
        ev, U = torch.linalg.eigh(S)
        K = K - U @ torch.diag_embed(F.relu(ev)) @ U.transpose(1, 2)
    if cap > 0:
        nrm = torch.linalg.matrix_norm(K).clamp_min(1e-300)
        K = K * torch.clamp(cap / nrm, max=1.0)[:, None, None]
    return K
 
 
def guard_forcing(g, z, stab, cap):
    """The matching safeguards for a forcing g (z_dot = ... + g): --stab removes the energy-increasing radial part
    (z^T g <= 0); --rate_cap scales so ||g|| <= cap ||z|| (the same energy-growth bound as for K)."""
    zz = (z * z).sum(-1, keepdim=True).clamp_min(1e-300)
    if stab:
        g = g - F.relu((z * g).sum(-1, keepdim=True)) / zz * z
    if cap > 0:
        ratio = (cap * zz.sqrt() / torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-300)).clamp(max=1.0)
        g = g * ratio
    return g
 
 
def psi_lower(X):
    """Lower triangle with halved diagonal: if M = L L^T then L^{-1} L_dot = psi_lower(L^{-1} M_dot L^{-T})."""
    return torch.tril(X, -1) + 0.5 * torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))
 
 
def transport_generator(Lc, H, transport=True):
    """N = L^{-1} H^T L^{-T} - L^{-1} L_dot with M_dot = H + H^T EXACTLY (exact Gram matrix): skew-symmetric.
    transport=False drops the first term (the d-form ablation d_dot = C d)."""
    Li_HT = torch.linalg.solve_triangular(Lc, H.transpose(1, 2), upper=False)
    A1 = torch.linalg.solve_triangular(Lc, Li_HT.transpose(1, 2), upper=False).transpose(1, 2)
    return (A1 if transport else 0.0) - psi_lower(A1 + A1.transpose(1, 2))
 
 
def cayley_apply(A, g, z, h):
    """Implicit midpoint for z' = A z + g, generator frozen: (I - h/2 A) z_new = (I + h/2 A) z + h g."""
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    rhs = ((I + h / 2 * A) @ z.unsqueeze(-1)).squeeze(-1) + (h * g if g is not None else 0.0)
    return solve(I - h / 2 * A, rhs.unsqueeze(-1)).squeeze(-1)
 
 
# ============================================================================= model
class EvolvingFA(nn.Module):
    def __init__(self, cfg, grid, scales):
        super().__init__()
        assert cfg.model in MODELS, cfg.model
        self.cfg, self.grid = cfg, grid
        self.u_s, self.ux_s, self.uxx_s, self.uxxx_s = scales
        L = grid.L
        self.basis = VonMisesBasis(cfg.G, L, cfg.w_min_frac * L, cfg.w_max_frac * L, cfg.basis)
        self.moving = cfg.model != "fixed"
        self.head = {"field_head": "field", "token_head": "token", "nosolve": "nosolve"}.get(cfg.model, "fa")
        self.transport = cfg.model != "no_transport"
        H, D = cfg.hidden, cfg.desc
        self.enc_feat = mlp([6, H, H])                                    # residual placement of the initial basis
        self.enc_out = mlp([2 * H, H, 2], last_zero=True)
        self.feat = mlp([6, H, H, H])                                     # point features of the decoded field
        self.to_q, self.to_k = nn.Linear(H, D), nn.Linear(H, D)           # FA / nosolve descriptors
        self.field_out = nn.Linear(H, 1)                                  # generic local head
        self.tok_in = nn.Linear(H + 1, H)                                 # generic nonlocal head
        self.tok_attn = nn.MultiheadAttention(H, 4, batch_first=True)
        self.tok_out = mlp([H, H, 1])
        self.log_lam = nn.Parameter(torch.tensor(math.log(cfg.lam)))
        self.fs = mlp([8, H, H])                                          # basis motion
        self.fs_out = mlp([2 * H, H, 2], last_zero=True)
        self.vel_scale, self.sig_rate = cfg.vel_scale, cfg.sig_rate
        self.jitter = [0]                                                 # Cholesky fallbacks (degenerate bases)
        self.register_buffer("x64", grid.x.double(), persistent=False)
 
    # ----- helpers
    def point_inputs(self, u, ux, uxx, uxxx):
        ang = TWO_PI * self.grid.x / self.grid.L
        B = u.shape[0]
        return torch.stack([u / self.u_s, ux / self.ux_s, uxx / self.uxx_s, uxxx / self.uxxx_s,
                            torch.sin(ang).expand(B, -1).to(u.dtype), torch.cos(ang).expand(B, -1).to(u.dtype)], -1)
 
    def geometry(self, theta, sig):
        """Basis and EXACT Gram matrix, in float64."""
        eta = torch.tanh(hi(sig))
        Phi = self.basis(hi(self.x64), hi(theta), eta)
        return Phi, self.grid.w * Phi.transpose(1, 2) @ Phi, eta
 
    def features(self, u):
        ders = self.grid.derivs(u)
        return self.feat(self.point_inputs(u, *ders)), ders
 
    def motion(self, theta, eta64, c64, u, ders, Phi):
        """Learned, speed-bounded basis velocity; returns theta_dot, sig_dot, H (float64), Phi_dot (float64)."""
        if not self.moving:
            self._rate = torch.zeros((), device=u.device)
            return torch.zeros_like(theta), torch.zeros_like(theta), None, None
        eta, c = eta64.to(u.dtype), c64.to(u.dtype)
        pos = self.basis.centre(theta)
        ux, uxx, uxxx = ders
        loc = torch.stack([torch.cos(theta), torch.sin(theta), eta, c / self.u_s,
                           interp_periodic(u, pos, self.grid) / self.u_s,
                           interp_periodic(ux, pos, self.grid) / self.ux_s,
                           interp_periodic(uxx, pos, self.grid) / self.uxx_s,
                           interp_periodic(uxxx, pos, self.grid) / self.uxxx_s], -1)
        h = self.fs(loc)
        out = self.fs_out(torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], -1))
        theta_dot = self.vel_scale * torch.tanh(out[..., 0])
        sig_dot = self.sig_rate * torch.tanh(out[..., 1])
        w = self.basis.width(eta)
        w_dot = (self.basis.w_max - self.basis.w_min) / 2 * (1 - eta ** 2) * sig_dot
        self._rate = ((theta_dot.abs() * self.grid.L / TWO_PI + w_dot.abs()) / w).amax().detach()
        _, Phi_dot = self.basis(hi(self.x64), hi(theta), eta64, hi(theta_dot), (1 - eta64 ** 2) * hi(sig_dot))
        return theta_dot, sig_dot, self.grid.w * Phi.transpose(1, 2) @ Phi_dot, Phi_dot
 
    def physics(self, Phi, M, Lc, c64, z64, feats):
        """Physics in Cholesky coordinates: z_dot_phys = K z + g. Returns (K, g), one of them None."""
        cfg, w = self.cfg, self.grid.w
        if self.head in ("fa", "nosolve"):
            Qt = w * Phi.transpose(1, 2) @ hi(self.to_q(feats))
            Kt = w * Phi.transpose(1, 2) @ hi(self.to_k(feats))
            lam = torch.exp(hi(self.log_lam))
            C = (fa_correspondence if self.head == "fa" else nosolve_correspondence)(Qt, Kt, M, lam)
            K = guard_generator(to_chol(C, Lc), cfg.stab, cfg.rate_cap)
            if getattr(cfg, "physics_explicit", False):
                return None, (K @ z64.unsqueeze(-1)).squeeze(-1)
            return K, None
        if self.head == "field":
            f = (w * Phi.transpose(1, 2) @ hi(self.field_out(feats))).squeeze(-1) * self.u_s
        else:
            mass = w * Phi.sum(1)                                                       # (B, G)
            tok = (w * Phi.transpose(1, 2) @ hi(feats)) / mass.clamp_min(1e-300).unsqueeze(-1)
            xt = self.tok_in(torch.cat([tok.to(feats.dtype), (c64 / self.u_s).to(feats.dtype).unsqueeze(-1)], -1))
            xt = xt + self.tok_attn(xt, xt, xt, need_weights=False)[0]
            f = hi(self.tok_out(xt).squeeze(-1)) * self.u_s * mass
        g = torch.linalg.solve_triangular(Lc, f.unsqueeze(-1), upper=False).squeeze(-1)
        return None, guard_forcing(g, z64, cfg.stab, cfg.rate_cap)
 
    # ----- encode
    def encode(self, u0):
        """Initial basis (even grid + bounded learned offsets) and dual coordinates d0 = Phi^T W u0 (float64).
        This is the field-metric fit: c0 = M^{-1} d0 minimises ||Phi c - u0||_W."""
        B, G = u0.shape[0], self.cfg.G
        theta_u = (TWO_PI * (torch.arange(G, device=u0.device, dtype=u0.dtype) + 0.5) / G).expand(B, -1)
        eta_u = torch.full((B, G), 2 * (0.6 * self.grid.L / G - self.basis.w_min) /
                           (self.basis.w_max - self.basis.w_min) - 1, device=u0.device, dtype=u0.dtype).clamp(-0.95, 0.95)
        if self.cfg.basis_init == "uniform":
            theta, eta = theta_u.clone(), eta_u
        else:
            ders = self.grid.derivs(u0)
            feats = self.enc_feat(self.point_inputs(u0, *ders))
            Phi_u = self.basis(self.grid.x, theta_u, eta_u)
            pooled = Phi_u.transpose(1, 2) @ feats / Phi_u.sum(1).clamp_min(1e-8).unsqueeze(-1)
            out = self.enc_out(torch.cat([pooled, pooled.mean(1, keepdim=True).expand_as(pooled)], -1))
            theta = theta_u + 0.45 * (TWO_PI / G) * torch.tanh(out[..., 0])     # < half a spacing: no crossings
            eta = torch.tanh(torch.atanh(eta_u) + 1.5 * torch.tanh(out[..., 1])).clamp(-0.98, 0.98)
        sig = torch.atanh(eta)
        Phi, _, _ = self.geometry(theta, sig)
        d64 = self.grid.w * (Phi.transpose(1, 2) @ hi(u0).unsqueeze(-1)).squeeze(-1)
        return theta, sig, d64
 
    # ----- Cholesky-coordinate formulation (default)
    def generator(self, theta, sig, z, aux=False):
        Phi, M, eta64 = self.geometry(theta, sig)
        Lc = chol(M, self.jitter)
        z64 = hi(z)
        c64 = torch.linalg.solve_triangular(Lc.transpose(1, 2), z64.unsqueeze(-1), upper=True).squeeze(-1)
        u = (Phi @ c64.unsqueeze(-1)).squeeze(-1).to(z.dtype)
        feats, ders = self.features(u)
        theta_dot, sig_dot, H, Phi_dot = self.motion(theta, eta64, c64, u, ders, Phi)
        A = transport_generator(Lc, H, self.transport) if H is not None else torch.zeros_like(M)
        K, g = self.physics(Phi, M, Lc, c64, z64, feats)
        if K is not None:
            A = A + K
        if aux:
            return theta_dot, sig_dot, A, g, dict(Phi=Phi, M=M, Lc=Lc, c=c64, u=u, K=K, g=g)
        return theta_dot, sig_dot, A, g
 
    def step_cayley(self, state, h):
        th, sg, z = state
        thd, sgd, A, g = self.generator(th, sg, z)
        z_mid = cayley_apply(A, g, hi(z), h / 2).to(z.dtype)
        thd_m, sgd_m, A_m, g_m = self.generator(th + h / 2 * thd, sg + h / 2 * sgd, z_mid)
        z_new = cayley_apply(A_m, g_m, hi(z), h).to(z.dtype)
        return th + h * thd_m, sg + h * sgd_m, z_new
 
    def to_z(self, theta, sig, d64):
        _, M, _ = self.geometry(theta, sig)
        return torch.linalg.solve_triangular(chol(M, self.jitter), hi(d64).unsqueeze(-1), upper=False).squeeze(-1)
 
    # ----- dual-coordinate formulation (RK4; kept for comparison and tests)
    def rhs(self, theta, sig, d, aux=False):
        Phi, M, eta64 = self.geometry(theta, sig)
        Lc = chol(M, self.jitter)
        c64 = solve(M, hi(d).unsqueeze(-1)).squeeze(-1)
        u = (Phi @ c64.unsqueeze(-1)).squeeze(-1).to(d.dtype)
        feats, ders = self.features(u)
        theta_dot, sig_dot, H, Phi_dot = self.motion(theta, eta64, c64, u, ders, Phi)
        z64 = (Lc.transpose(1, 2) @ c64.unsqueeze(-1)).squeeze(-1)                     # z = L^{-1} d = L^T c
        K, g = self.physics(Phi, M, Lc, c64, z64, feats)
        phys_z = (K @ z64.unsqueeze(-1)).squeeze(-1) if K is not None else g
        phys = (Lc @ phys_z.unsqueeze(-1)).squeeze(-1)                                # back to d coordinates
        d_dot = phys
        if H is not None and self.transport:
            d_dot = d_dot + (H.transpose(1, 2) @ c64.unsqueeze(-1)).squeeze(-1)        # H^T M^{-1} d = H^T c
        if aux:
            return (theta_dot, sig_dot, d_dot.to(d.dtype)), dict(Phi=Phi, Phi_dot=Phi_dot, M=M, H=H, c=c64, u=u,
                                                                 phys=phys)
        return theta_dot, sig_dot, d_dot.to(d.dtype)
 
    def rk4(self, state, h):
        k1 = self.rhs(*state)
        k2 = self.rhs(*[s + h / 2 * k for s, k in zip(state, k1)])
        k3 = self.rhs(*[s + h / 2 * k for s, k in zip(state, k2)])
        k4 = self.rhs(*[s + h * k for s, k in zip(state, k3)])
        return tuple(s + h / 6 * (a + 2 * b + 2 * cc + e) for s, a, b, cc, e in zip(state, k1, k2, k3, k4))
 
    # ----- decoding and rollout
    def decode_state(self, state, x=None):
        """Decoded field (state dtype), exact Gram M, coefficients c, basis Phi (float64), field energy 1/2||u||_W^2."""
        th, sg, v = state
        Phi, M, eta64 = self.geometry(th, sg)
        if self.cfg.integrator == "cayley":
            Lc = chol(M, self.jitter)
            c = torch.linalg.solve_triangular(Lc.transpose(1, 2), hi(v).unsqueeze(-1), upper=True).squeeze(-1)
        else:
            c = solve(M, hi(v).unsqueeze(-1)).squeeze(-1)
        u64 = (Phi @ c.unsqueeze(-1)).squeeze(-1)
        energy = 0.5 * self.grid.w * (u64 ** 2).sum(-1)
        if x is not None:
            u64 = (self.basis(hi(x), hi(th), eta64) @ c.unsqueeze(-1)).squeeze(-1)
        return u64.to(v.dtype), M, c, Phi, energy
 
    def rollout(self, u0, t_query, dt, keep=False):
        """Integrate from t=0 to each sorted query time. Step size adapts CFL-style for accuracy:
        h * (bump speed / bump width + rate cap) <= cfl."""
        theta, sig, d64 = self.encode(u0)
        cayley = self.cfg.integrator == "cayley"
        state = (theta, sig, (self.to_z(theta, sig, d64) if cayley else d64).to(u0.dtype))
        with torch.no_grad():
            self.generator(*state) if cayley else self.rhs(*state)
        t_prev, outs, Ms, states, evals = 0.0, [], [], [], 1
        for tq in t_query:
            span = float(tq) - t_prev
            base = max(1, math.ceil(span / dt - 1e-9)) if span > 0 else 0
            rate = float(self._rate) + max(self.cfg.rate_cap, 0.0)
            n_sub = min(max(base, math.ceil(span * rate / self.cfg.cfl - 1e-9)), self.cfg.max_sub * base) if span > 0 else 0
            for _ in range(n_sub):
                state = self.step_cayley(state, span / n_sub) if cayley else self.rk4(state, span / n_sub)
                evals += 2 if cayley else 4
            u, M, _, _, _ = self.decode_state(state)
            outs.append(u)
            Ms.append(M)
            if keep:
                states.append(tuple(s.detach() for s in state))
            t_prev = float(tq)
        return torch.stack(outs, 1), torch.stack(Ms, 1), states, evals
 
 
# ============================================================================= losses and diagnostics
def rel_l2(pred, true):
    return torch.linalg.norm(pred - true, dim=-1) / (torch.linalg.norm(true, dim=-1) + 1e-8)
 
 
def nondegeneracy_penalty(Ms, floor):
    """Encourage (not enforce) a well-conditioned basis: penalise log of the smallest eigenvalue of the
    unit-diagonal Gram matrix below log(floor)."""
    dg = Ms.diagonal(dim1=-2, dim2=-1).clamp_min(1e-300)
    Mn = Ms / torch.sqrt(dg[..., :, None] * dg[..., None, :])
    lmin = torch.linalg.eigvalsh(Mn)[..., 0].clamp_min(1e-12)
    return F.relu(math.log(floor) - torch.log(lmin)).mean().float()
 
 
@torch.no_grad()
def span_diagnostics(Phi, u, w, rcond=1e-10):
    """Best relative L2 error of u in span(Phi) under W, by rank-aware SVD least squares in float64.
    Returns (error, numerical rank, smallest/largest singular value)."""
    A, b = math.sqrt(w) * hi(Phi), math.sqrt(w) * hi(u)
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    mask = (S > rcond * S[:, :1]).to(A.dtype)
    proj = U @ ((U.transpose(1, 2) @ b.unsqueeze(-1)) * mask.unsqueeze(-1))
    err = torch.linalg.norm(proj.squeeze(-1) - b, dim=-1) / torch.linalg.norm(b, dim=-1).clamp_min(1e-300)
    return err.float(), mask.sum(-1).long(), (S[:, -1] / S[:, 0]).float()
 
 
@torch.no_grad()
def max_subspace_angle(Phi1, Phi2, rcond=1e-10):
    """Largest principal angle between two spans (W uniform), using rank-aware orthonormal bases from the SVD.
    Spans of different numerical rank are reported as pi/2."""
    U1, S1, _ = torch.linalg.svd(hi(Phi1), full_matrices=False)
    U2, S2, _ = torch.linalg.svd(hi(Phi2), full_matrices=False)
    m1, m2 = S1 > rcond * S1[:, :1], S2 > rcond * S2[:, :1]
    r1, r2 = m1.sum(-1), m2.sum(-1)
    s = torch.linalg.svdvals((U1 * m1.unsqueeze(1)).transpose(1, 2) @ (U2 * m2.unsqueeze(1)))   # descending
    k = torch.minimum(r1, r2).clamp(min=1) - 1
    sk = torch.gather(s, 1, k.unsqueeze(-1)).squeeze(-1).clamp(-1, 1)
    return torch.where(r1 == r2, torch.arccos(sk), torch.full_like(sk, math.pi / 2)).float()
 
 
# ============================================================================= tests
def run_tests():
    torch.manual_seed(0)
    dt64, ok = torch.float64, True
 
    def check(name, err, tol):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name:70s} {err:.2e}")
 
    L, n, G, B = TWO_PI, 128, 5, 2
    x = torch.arange(n, dtype=dt64) * L / n
    grid = Grid(x, L)
 
    class Cfg:
        G = 5; basis = "bump"; w_min_frac = 0.01; w_max_frac = 0.25; model = "full"; hidden = 32; desc = 16
        lam = 0.1; vel_scale = 1.0; sig_rate = 1.0; basis_init = "residual"; stab = False; rate_cap = 0.0
        cfl = 1.0; max_sub = 8; integrator = "cayley"; physics_explicit = False
 
    # 1. basis: analytic time derivative, and smooth periodicity (spectral vs analytic x-derivative), incl. max width
    for kind in ("bump", "pou"):
        basis = VonMisesBasis(G, L, 0.05, 1.5, kind)
        th = torch.rand(B, G, dtype=dt64) * TWO_PI
        et = torch.tensor([[-0.9, -0.4, 0.0, 0.5, 0.999]], dtype=dt64).expand(B, -1)
        thd, etd = torch.randn(B, G, dtype=dt64), torch.randn(B, G, dtype=dt64)
        h = 1e-6
        fd = (basis(x, th + h * thd, et + h * etd) - basis(x, th - h * thd, et - h * etd)) / (2 * h)
        _, an = basis(x, th, et, thd, etd)
        check(f"basis '{kind}': analytic Phi_dot vs finite difference", ((an - fd).abs().max() / fd.abs().max()).item(), 1e-7)
        et_w = torch.tensor([[0.3, 0.5, 0.7, 0.9, 0.999]], dtype=dt64).expand(B, -1)   # wide bumps: where a kink
        P = basis(x, th, et_w)                                                           # would show (old basis)
        spec = grid.derivs(P.transpose(1, 2))[0].transpose(1, 2)
        ana = basis.dx(x, th, et_w)
        check(f"basis '{kind}': smooth and periodic, wide bumps (spectral = analytic x-derivative)",
              ((spec - ana).abs().max() / ana.abs().max()).item(), 1e-8)
 
    # 2. FA solve covariant with lam*M ridge; the nosolve ablation is not
    Phi = torch.randn(B, n, G, dtype=dt64)
    FQ, FK = torch.randn(B, n, 16, dtype=dt64), torch.randn(B, n, 16, dtype=dt64)
    A = torch.eye(G, dtype=dt64) + 0.4 * torch.randn(B, G, G, dtype=dt64)
    ip = lambda P, Fm: grid.w * P.transpose(1, 2) @ Fm
    M, Mt = ip(Phi, Phi), ip(Phi @ A, Phi @ A)
    C = fa_correspondence(ip(Phi, FQ), ip(Phi, FK), M, 0.3)
    Ct = fa_correspondence(ip(Phi @ A, FQ), ip(Phi @ A, FK), Mt, 0.3)
    AinvT = torch.linalg.inv(A.transpose(1, 2))
    check("FA solve: C~ = A^T C A^-T with lam*M ridge", (Ct - A.transpose(1, 2) @ C @ AinvT).abs().max().item(), 1e-9)
    Cn = nosolve_correspondence(ip(Phi, FQ), ip(Phi, FK), M, 0.3)
    Cnt = nosolve_correspondence(ip(Phi @ A, FQ), ip(Phi @ A, FK), Mt, 0.3)
    print(f"[INFO] nosolve ablation covariance error (expected nonzero): "
          f"{(Cnt - A.transpose(1, 2) @ Cn @ AinvT).abs().max().item():.2e}")
    for stab, cap, name in ((True, 0.0, "stability projection"), (False, 0.5, "rate cap")):
        Lc, Lct = torch.linalg.cholesky(M), torch.linalg.cholesky(Mt)
        Cs = from_chol(guard_generator(to_chol(C, Lc), stab, cap), Lc)
        Cst = from_chol(guard_generator(to_chol(Ct, Lct), stab, cap), Lct)
        check(f"{name}: frame covariant", (Cst - A.transpose(1, 2) @ Cs @ AinvT).abs().max().item(), 1e-9)
    Ks = guard_generator(to_chol(C, torch.linalg.cholesky(M)), True, 0.5)
    check("stab + cap: max eig sym(K) <= 0", max(torch.linalg.eigvalsh(0.5 * (Ks + Ks.transpose(1, 2))).max().item(), 0), 1e-12)
    check("stab + cap: ||K||_F <= cap", max((torch.linalg.matrix_norm(Ks) - 0.5).max().item(), 0), 1e-12)
    zr, gr = torch.randn(B, G, dtype=dt64), torch.randn(B, G, dtype=dt64)
    gg = guard_forcing(gr, zr, True, 0.5)
    check("forcing guard: z^T g <= 0 and ||g|| <= cap ||z||",
          max((zr * gg).sum(-1).max().item(), (gg.norm(dim=-1) - 0.5 * zr.norm(dim=-1)).max().item(), 0), 1e-12)
 
    # 3. model identities with the EXACT Gram matrix (d-form): energy identity, decomposition, orthogonality,
    #    and the reported energy equals the decoded field energy exactly (z-form)
    for stab in (False, True):
        cfg = Cfg(); cfg.stab = stab
        model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0, 1.0)).double()
        with torch.no_grad():
            for p_ in model.fs_out.parameters():
                p_.normal_(0, 0.3)
            theta = torch.rand(B, G, dtype=dt64) * TWO_PI
            sig = torch.atanh(torch.rand(B, G, dtype=dt64) * 1.2 - 0.6)
            d = torch.randn(B, G, dtype=dt64)
            (thd, sgd, dd), a = model.rhs(theta, sig, d, aux=True)
        c, Mm, Hm = a["c"], a["M"], a["H"]
        cond = torch.linalg.cond(Mm).max().item()
        tol = max(1e-9, 1e-12 * cond)
        Mdot = Hm + Hm.transpose(1, 2)
        t1, t2 = (c * dd).sum(-1), 0.5 * (c.unsqueeze(1) @ Mdot @ c.unsqueeze(-1)).squeeze()
        target = (c * a["phys"]).sum(-1)
        check(f"energy identity E' = c^T phys, relative (stab={stab}, cond M {cond:.0e})",
              ((t1 - t2 - target).abs() / (t1.abs() + t2.abs() + target.abs() + 1e-300)).max().item(), tol)
        if stab:
            check("with stab: E' <= 0 (relative)", max(((t1 - t2) / (t1.abs() + t2.abs())).max().item(), 0), tol)
        c_dot = solve(Mm, (dd - (Mdot @ c.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        u_dot = a["Phi_dot"] @ c.unsqueeze(-1) + a["Phi"] @ c_dot.unsqueeze(-1)
        out_span = a["Phi_dot"] @ c.unsqueeze(-1) - a["Phi"] @ solve(Mm, Hm @ c.unsqueeze(-1))
        in_span = a["Phi"] @ solve(Mm, a["phys"].unsqueeze(-1))
        check(f"velocity decomposition (I-P)Phi_dot c + Phi M^-1 phys (stab={stab})",
              ((u_dot - out_span - in_span).abs().max() / u_dot.abs().max()).item(), tol)
        ipw = lambda f, g: grid.w * (f * g).sum((1, 2))
        cosine = ipw(out_span, in_span) / torch.sqrt(ipw(out_span, out_span) * ipw(in_span, in_span))
        check(f"decomposition parts W-orthogonal, cosine (stab={stab})", cosine.abs().max().item(), tol)
        with torch.no_grad():
            z = model.to_z(theta, sig, d)
            _, _, _, _, energy = model.decode_state((theta, sig, z))
        check(f"reported energy 1/2||z||^2 = field energy 1/2||u||_W^2 (exact Gram, stab={stab})",
              ((0.5 * (z ** 2).sum(-1) - energy).abs() / energy).max().item(), 1e-10)
 
    # 4. Cayley integrator: transport skew; energy non-increasing for linear heads with stab, for any step;
    #    explicit heads satisfy the continuous-time guards
    Phi = torch.randn(B, n, G, dtype=dt64); Phid = torch.randn(B, n, G, dtype=dt64)
    Nz = transport_generator(torch.linalg.cholesky(ip(Phi, Phi)), ip(Phi, Phid))
    check("transport generator N is skew (relative)", ((Nz + Nz.transpose(1, 2)).abs().max() / Nz.abs().max()).item(), 1e-12)
    for head in ("full", "nosolve"):
        cfg = Cfg(); cfg.model = head; cfg.stab = True
        model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0, 1.0)).double()
        with torch.no_grad():
            for p_ in model.fs_out.parameters():
                p_.normal_(0, 0.3)
            st = (torch.rand(B, G, dtype=dt64) * TWO_PI, torch.atanh(torch.rand(B, G, dtype=dt64) * 1.2 - 0.6),
                  torch.randn(B, G, dtype=dt64))
            Es = [0.5 * (st[2] ** 2).sum(-1)]
            for _ in range(10):
                st = model.step_cayley(st, 0.5)
                Es.append(0.5 * (st[2] ** 2).sum(-1))
            E = torch.stack(Es, 1)
        check(f"Cayley + stab ({head}): energy non-increasing, 10 steps of h=0.5",
              max(((E[:, 1:] - E[:, :-1]) / E[:, :-1]).max().item(), 0), 1e-10)
    for head in ("field_head", "token_head"):
        cfg = Cfg(); cfg.model = head; cfg.stab = True; cfg.rate_cap = 0.7
        model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0, 1.0)).double()
        with torch.no_grad():
            st = (torch.rand(B, G, dtype=dt64) * TWO_PI, torch.atanh(torch.rand(B, G, dtype=dt64) * 1.2 - 0.6),
                  torch.randn(B, G, dtype=dt64))
            _, _, _, g, a = model.generator(*st, aux=True)
        check(f"{head}: forcing guards z^T g <= 0, ||g|| <= cap ||z||",
              max((st[2] * g).sum(-1).max().item(), (g.norm(dim=-1) - 0.7 * st[2].norm(dim=-1)).max().item(), 0), 1e-12)
 
    # 5. the z-form (Cayley) converges to the d-form (RK4) at second order
    cfg = Cfg(); cfg.rate_cap = 5.0
    model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0, 1.0)).double()
    with torch.no_grad():
        for p_ in model.fs_out.parameters():
            p_.normal_(0, 0.1)
        th0 = (TWO_PI * (torch.arange(G, dtype=dt64) + 0.5) / G).expand(B, -1) + 0.1 * torch.randn(B, G, dtype=dt64)
        sg0 = torch.atanh(torch.rand(B, G, dtype=dt64) * 0.6 - 0.6)
        u_smooth = (torch.sin(x) + 0.5 * torch.cos(2 * x)).expand(B, -1)
        Phi0, _, _ = model.geometry(th0, sg0)
        d0 = grid.w * (Phi0.transpose(1, 2) @ u_smooth.unsqueeze(-1)).squeeze(-1)
        T = 0.2
        s_ref = (th0, sg0, d0)
        for _ in range(400):
            s_ref = model.rk4(s_ref, T / 400)
        model.cfg.integrator = "rk4"
        u_ref = model.decode_state(s_ref)[0]
        model.cfg.integrator = "cayley"
        errs = []
        for n_steps in (10, 20, 40):
            s = (th0, sg0, model.to_z(th0, sg0, d0))
            for _ in range(n_steps):
                s = model.step_cayley(s, T / n_steps)
            errs.append(((model.decode_state(s)[0] - u_ref).abs().max() / u_ref.abs().max()).item())
    ratios = [a_ / b_ for a_, b_ in zip(errs, errs[1:])]
    print(f"       Cayley vs fine RK4: errors {', '.join(f'{e:.1e}' for e in errs)}; ratios "
          f"{', '.join(f'{r:.2f}' for r in ratios)} (4 = second order)")
    check("Cayley z-form converges to RK4 d-form at second order (3 - min ratio)", max(3.0 - min(ratios), 0.0), 0.0)
    check("Cayley z-form matches RK4 d-form at the finest step (relative)", errs[-1], 1e-3)
 
    # 6. diagnostics: exact for in-span targets on a nearly degenerate basis; angles frame-invariant; rank loss
    xs = torch.arange(n, dtype=dt64) * 40.0 / n
    mus = (torch.arange(G, dtype=dt64) + 0.5) * 40.0 / G
    mus[1] = mus[0] + 1e-4
    Pd = torch.exp(-(((xs[:, None] - mus + 20) % 40 - 20) ** 2) / 18.0).unsqueeze(0)
    u_in = (Pd @ torch.randn(1, G, 1, dtype=dt64)).squeeze(-1)
    err, rank, smin = span_diagnostics(Pd, u_in, 40.0 / n)
    check(f"span diagnostic: in-span target on cond~1e10 basis (rank {rank.item()}/{G})", err.item(), 1e-10)
    Ar = torch.eye(G, dtype=dt64) + 0.3 * torch.randn(1, G, G, dtype=dt64)
    check("subspace angle: span vs reframed copy of itself", max_subspace_angle(Pd, Pd @ Ar).item(), 1e-6)
    Pr = torch.cat([Pd[..., :1], Pd[..., :1], Pd[..., 2:]], -1)                    # exact duplicate column
    _, rk, _ = span_diagnostics(Pr, u_in, 40.0 / n)
    check("span diagnostic detects an exact rank loss", float(rk.item() != G - 1), 0.0)
    check("subspace angle: rank-deficient span vs itself is finite and ~0",
          max_subspace_angle(Pr, Pr).item() if math.isfinite(max_subspace_angle(Pr, Pr).item()) else 1.0, 1e-6)
 
    # 7. no temporal leakage: preprocessing reads only allowed snapshots (others are NaN here)
    tt = torch.arange(21, dtype=torch.float32) * 0.1
    uu = torch.sin(x.float()[None, None, :] - tt[None, :, None]).expand(4, -1, -1).clone()
    seen, unseen, extrap = time_splits(tt, 1.0)
    forbidden = unseen + extrap
    uu[:, forbidden] = float("nan")
    sc = normalisation_scales(uu, [0] + seen, Grid(x.float(), L))
    sp = estimate_max_speed(uu, tt, L, [0] + seen, 1.0)
    check("normalisation and speed use only training snapshots (finite with others NaN)",
          float(not (all(math.isfinite(v) for v in sc) and math.isfinite(sp))), 0.0)
 
    # 8. smoke: every variant, both integrators, forward + backward in float32 (geometry in float64)
    xs32 = torch.arange(64, dtype=torch.float32) * L / 64
    g32 = Grid(xs32, L)
    u0 = torch.exp(-((xs32 - 2.0) ** 2) / 0.1).expand(2, -1).contiguous()
    runs = [(m, i, False) for i in ("cayley", "rk4") for m in MODELS] + [("full", "cayley", True)]
    for variant, integ, expl in runs:
        cfg = Cfg(); cfg.model = variant; cfg.integrator = integ; cfg.rate_cap = 3.0; cfg.physics_explicit = expl
        m = EvolvingFA(cfg, g32, (0.5, 2.0, 10.0, 50.0))
        pred, Ms, _, ev = m.rollout(u0, [0.1, 0.2], dt=0.05)
        loss = rel_l2(pred, u0.unsqueeze(1).expand_as(pred)).mean() + 0.01 * nondegeneracy_penalty(Ms, 1e-3)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        ok &= finite
        tag = f"{variant}{' (explicit)' if expl else ''}"
        print(f"[{'PASS' if finite else 'FAIL'}] {tag:24s} {integ:6s} rollout + backward ({ev} evals), loss {loss.item():.3f}")
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return ok
 
 
# ============================================================================= train / evaluate
@torch.no_grad()
def evaluate(model, data, cfg, splits, max_traj=None):
    model.eval()
    seen, unseen, extrap = splits
    allq = sorted(seen + unseen + extrap)
    tq = data.t[allq].tolist()
    errs, proj, angles, ranks_lost, smins, energy, centres, rt, evals_total = [], [], [], 0, [], [], [], 0.0, 0
    jit0 = model.jitter[0]
    N = data.u.shape[0] if max_traj is None else min(max_traj, data.u.shape[0])
    n_batches = max(1, math.ceil(N / cfg.batch_eval))
    for i in range(0, N, cfg.batch_eval):
        u = data.u[i:i + cfg.batch_eval]
        if u.is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        pred, _, states, ev = model.rollout(u[:, 0], tq, cfg.dt, keep=True)
        if u.is_cuda:
            torch.cuda.synchronize()
        rt += time.time() - t0
        evals_total += ev
        true = u[:, allq]
        errs.append(rel_l2(pred, true))
        p_err, ang, en, cen, sm, prev = [], [], [], [], [], None
        for k, st in enumerate(states):
            _, _, _, Phi, e = model.decode_state(st)
            pe, rk, smin = span_diagnostics(Phi, true[:, k], model.grid.w)
            ranks_lost += int((rk < cfg.G).sum())
            p_err.append(pe); sm.append(smin)
            ang.append(torch.zeros(u.shape[0], device=u.device) if prev is None else max_subspace_angle(prev, Phi))
            en.append(e.float())
            cen.append(model.basis.centre(st[0]))
            prev = Phi
        proj.append(torch.stack(p_err, 1)); angles.append(torch.stack(ang, 1)); smins.append(torch.stack(sm, 1))
        energy.append(torch.stack(en, 1)); centres.append(torch.stack(cen, 1))
    errs, proj = torch.cat(errs).cpu(), torch.cat(proj).cpu()
    pos = {q: j for j, q in enumerate(allq)}
    summary = {name: errs[:, [pos[q] for q in qs]].mean().item() if qs else float("nan")
               for name, qs in (("seen", seen), ("unseen", unseen), ("extrapolation", extrap))}
    summary.update(runtime_per_rollout_s=rt / n_batches, rhs_evals_per_rollout=evals_total / n_batches,
                   rank_deficient_states=ranks_lost, cholesky_fallbacks=model.jitter[0] - jit0,
                   min_singular_ratio=torch.cat(smins).min().item())
    model.train()
    return dict(summary=summary, t=data.t[allq].tolist(),
                err_vs_t=errs.mean(0).tolist(), proj_err_vs_t=proj.mean(0).tolist(),
                subspace_angle_vs_t=torch.cat(angles).cpu().mean(0).tolist(),
                energy_example=torch.cat(energy)[0].cpu().tolist(),
                centres_example=torch.cat(centres)[0].cpu().tolist())
 
 
def plot(result, data, model, cfg, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.array(result["t"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(t, result["err_vs_t"], label="forecast error")
    ax[0].plot(t, result["proj_err_vs_t"], "--", label="best error in current span")
    ax[0].axvline(cfg.T_train, c="gray", ls=":", label="end of training horizon")
    ax[0].set(yscale="log", xlabel="t", ylabel="relative L2", title=f"{cfg.problem} / {cfg.model} / G={cfg.G}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, result["subspace_angle_vs_t"])
    ax[1].set(xlabel="t", ylabel="max principal angle (rad)", title="subspace motion between query times")
    ax[1].grid(alpha=0.3)
    with torch.no_grad():
        u = data.u[:1]
        pred, _, states, _ = model.rollout(u[:, 0], data.t[1:].tolist(), cfg.dt, keep=True)
    x = data.x.cpu().numpy()
    err = (pred[0] - u[0, 1:]).abs().cpu().numpy()
    ax[2].imshow(err, aspect="auto", origin="lower", cmap="magma", extent=[x[0], x[-1], float(data.t[1]), float(data.t[-1])])
    cen = torch.stack([model.basis.centre(s[0])[0] for s in states]).cpu().numpy()
    for g in range(cen.shape[1]):
        ax[2].scatter(cen[:, g], data.t[1:].numpy(), s=1, c="cyan")
    ax[2].axhline(cfg.T_train, c="white", ls=":")
    ax[2].set(xlabel="x", ylabel="t", title="|error| on test trajectory 0, basis centres in cyan")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results.png"), dpi=130)
    plt.close(fig)
 
 
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true")
    p.add_argument("--problem", choices=list(PROBLEM_DEFAULTS), default="kdv")
    p.add_argument("--model", choices=MODELS, default="full")
    p.add_argument("--physics_explicit", action="store_true",
                   help="integrate FA / nosolve physics explicitly, as the generic heads must be")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--basis", choices=["bump", "pou"], default=None)
    p.add_argument("--basis_init", choices=["residual", "uniform"], default="residual")
    p.add_argument("--w_min_frac", type=float, default=None)
    p.add_argument("--w_max_frac", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--desc", type=int, default=32, help="FA descriptor channels D")
    p.add_argument("--lam", type=float, default=0.1, help="initial FA ridge (learned)")
    p.add_argument("--stab", action="store_true", help="energy non-increasing (every head)")
    p.add_argument("--rate_cap", type=float, default=None, help="energy growth rate bound, every head (default 10/T_train)")
    p.add_argument("--vel_scale", type=float, default=None, help="max basis angular speed (default 2x data speed)")
    p.add_argument("--sig_rate", type=float, default=None, help="max rate of the width variable (default 4/T_train)")
    p.add_argument("--integrator", choices=["cayley", "rk4"], default="cayley")
    p.add_argument("--cfl", type=float, default=1.0)
    p.add_argument("--max_sub", type=int, default=8)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--stride", type=int, default=2, help="quadrature = every stride-th grid point")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--warm", type=int, default=1000)
    p.add_argument("--T_start_frac", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--batch_eval", type=int, default=50)
    p.add_argument("--n_query", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg_nondeg", type=float, default=0.1)
    p.add_argument("--nondeg_floor", type=float, default=1e-2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--max_eval_traj", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cfg = p.parse_args()
    if cfg.test:
        raise SystemExit(0 if run_tests() else 1)
 
    defaults = PROBLEM_DEFAULTS[cfg.problem]
    cfg.basis = cfg.basis or defaults["basis"]
    cfg.dt = cfg.dt or defaults["dt"]
    cfg.w_min_frac = cfg.w_min_frac or defaults["w_min_frac"]
    tag = f"{cfg.model}{'_explicit' if cfg.physics_explicit else ''}"
    cfg.out = cfg.out or f"runs/{cfg.problem}_{tag}_G{cfg.G}_s{cfg.seed}"
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
 
    train = Trajectories(os.path.join(cfg.data, f"{cfg.problem}_train.npz"), cfg.stride, cfg.device)
    test = Trajectories(os.path.join(cfg.data, f"{cfg.problem}_test.npz"), cfg.stride, cfg.device)
    cfg.T_train = float(train.t[-1]) * defaults["train_frac"]
    splits = time_splits(train.t, cfg.T_train)
    allowed = [0] + splits[0]                                   # t = 0 and the seen times only
    grid = Grid(train.x, train.L)
    scales = normalisation_scales(train.u, allowed, grid)
    speed = estimate_max_speed(train.u, train.t, train.L, allowed, cfg.T_train)
    cfg.vel_scale = cfg.vel_scale or 2.0 * speed * TWO_PI / train.L
    cfg.sig_rate = cfg.sig_rate or 4.0 / cfg.T_train
    cfg.rate_cap = 10.0 / cfg.T_train if cfg.rate_cap is None else cfg.rate_cap
 
    model = EvolvingFA(cfg, grid, scales).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"=== {cfg.problem} / {tag} / basis {cfg.basis} / G={cfg.G} / seed {cfg.seed} / {n_params:,} params / "
          f"T_train={cfg.T_train} / dt={cfg.dt} / quadrature n={grid.n}")
    print(f"    structure speed ~{speed:.3f} -> basis speed limit {cfg.vel_scale * train.L / TWO_PI:.3f}; "
          f"narrowest bump {model.basis.w_min:.3f}; rate cap {cfg.rate_cap:.3g}")
 
    seen_t = train.t[splits[0]]
    history, t0, skipped, window = [], time.time(), 0, []
    for it in range(1, cfg.iters + 1):
        T_cur = cfg.T_train * (cfg.T_start_frac + (1 - cfg.T_start_frac) * min(1.0, it / max(1, cfg.warm)))
        avail = [i for i, tt in zip(splits[0], seen_t.tolist()) if tt <= T_cur + 1e-9]
        q = sorted(np.random.choice(avail, size=min(cfg.n_query, len(avail)), replace=False).tolist())
        b = torch.randint(0, train.u.shape[0], (cfg.batch,), device=cfg.device)
        u = train.u[b]
        pred, Ms, _, _ = model.rollout(u[:, 0], train.t[q].tolist(), cfg.dt)
        loss = rel_l2(pred, u[:, q]).mean()
        opt.zero_grad()
        reg = torch.tensor(float("nan"))
        if torch.isfinite(loss) and torch.isfinite(Ms).all():
            try:
                reg = nondegeneracy_penalty(Ms, cfg.nondeg_floor)
            except RuntimeError:
                pass
        if not (torch.isfinite(loss) and torch.isfinite(reg)) or loss.item() > 1e3:
            skipped += 1
            sched.step()
            if skipped > 0.2 * it and it >= 50:
                raise RuntimeError(f"{skipped} of {it} batches non-finite or absurd: try --stab, smaller --dt or --lr.")
            continue
        (loss + cfg.reg_nondeg * reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        window.append(loss.item())
        if it % cfg.log_every == 0:
            print(f"  it {it:5d}  horizon {T_cur:.2f}  train rel-L2 mean {np.mean(window) if window else float('nan'):.4f}"
                  f" median {np.median(window) if window else float('nan'):.4f}  nondeg {reg.item():.3f}  "
                  f"skipped {skipped}  chol-fallbacks {model.jitter[0]}  "
                  f"({(time.time() - t0) / it:.2f} s/it, {time.time() - t0:.0f}s)")
            window = []
        if it % cfg.eval_every == 0 or it == cfg.iters:
            res = evaluate(model, test, cfg, splits, max_traj=cfg.max_eval_traj)
            s = res["summary"]
            print(f"  [eval] seen {s['seen']:.4f}  unseen {s['unseen']:.4f}  extrapolation {s['extrapolation']:.4f}"
                  f"  | {s['runtime_per_rollout_s']:.2f}s/batch, {s['rhs_evals_per_rollout']:.0f} evals"
                  f"  | rank-deficient states {s['rank_deficient_states']}, chol fallbacks {s['cholesky_fallbacks']}")
            history.append(dict(it=it, **s))
 
    res["history"] = history
    res["config"] = {k: v for k, v in vars(cfg).items()}
    res["n_params"] = n_params
    with open(os.path.join(cfg.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save(model.state_dict(), os.path.join(cfg.out, "model.pt"))
    plot(res, test, model, cfg, cfg.out)
    print(f"saved {cfg.out}/results.json, model.pt, results.png")
 
 
if __name__ == "__main__":
    main()