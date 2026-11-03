"""
Evolving-basis Functional Attention dynamics (prototype v1).

Model.  The field is u(x,t) = Phi(x; s(t)) c(t): G moving bumps (centres and widths in the basis state s)
with coefficients c. The state integrated in time is (s, d), with d = Phi^T W u the dual coordinates.

    d_dot = C_theta(u) d  +  H^T c                  H = Phi^T W Phi_dot,   c = M^{-1} d,   M = Phi^T W Phi
            ~~~~~~~~~~~~     ~~~~~~
            physics: FA      basis motion: exact geometric transport (continuous FA correspondence)
    C_theta = Q~ K~^T (K~ K~^T + lam M)^{-1},   Q~ = Phi^T W F_Q(u),  K~ = Phi^T W F_K(u)
    s_dot   = f_s(s, d)                         learned, bounded speeds; widths via tanh, valid for any state

Decoded velocity  u_dot = (I - P) Phi_dot c  +  Phi M^{-1} C_theta d   (out-of-span motion + in-span physics).

Integration (--integrator cayley, default). In Cholesky coordinates z = L^{-1} d (M = L L^T) the field energy is
exactly 1/2 ||z||^2 and the dynamics read z' = (K + N) z, with K = L^{-1} C_theta L the physics and
N = L^{-1} H^T L^{-T} - L^{-1} L_dot the basis motion, which is exactly skew. Each step: explicit midpoint for the
basis state, implicit midpoint (a Cayley transform) for z with the generator frozen at the midpoint. Then
||z_{n+1}||^2 - ||z_n||^2 = 2h zbar^T sym(K) zbar for ANY step size h: crossing bumps (where M is nearly singular
and N becomes huge) cannot inject energy, which is what makes classical RK4 blow up on this model.

Variants (--model):
    full          moving basis, FA physics, transport term          (the proposed model)
    fixed         basis placed at t=0 by the encoder, then frozen    (does evolving the subspace help?)
    no_transport  moving basis, FA physics, transport term removed   (does the geometric term help?)
    field_head    moving basis, generic Galerkin head Phi^T W F(u)   (does FA's solve add value?)

Problems (from fa_problems.py): advect, kdv, burgers. Training is autonomous rollout from u(t=0); query
times are sampled from the even-indexed snapshots up to T_train. Evaluation reports three regimes:
seen times (even, <= T_train), unseen times (odd, <= T_train), and extrapolation (t > T_train).

Usage:
    python fa_evolve.py --test
    python fa_evolve.py --problem kdv --model full --data data/ --out runs/kdv_full
    python fa_evolve.py --problem kdv --model fixed --data data/ --out runs/kdv_fixed
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
PROBLEM_DEFAULTS = {             # basis family, time step, training horizon as a fraction of the data
    "advect":  dict(basis="gauss", dt=0.025, train_frac=0.5, w_min_frac=0.01),   # pulses: sigma >= 0.0127 L
    "kdv":     dict(basis="gauss", dt=0.1,   train_frac=0.5, w_min_frac=0.02),   # solitons: sigma >= ~0.03 L
    "burgers": dict(basis="pou",   dt=0.025, train_frac=0.5, w_min_frac=0.01),
}


# ============================================================================= data
class Trajectories:
    def __init__(self, path, stride, device):
        d = np.load(path)
        self.u = torch.tensor(d["u"][:, :, ::stride], dtype=torch.float32, device=device)   # (N, T, n)
        self.t = torch.tensor(d["t"], dtype=torch.float32)
        self.x = torch.tensor(d["x"][::stride], dtype=torch.float32, device=device)
        self.L = float(d["L"])


class Grid:
    """Uniform periodic quadrature grid: weights W = L/n, spectral derivatives."""

    def __init__(self, x, L):
        self.x, self.L, self.n = x, L, x.numel()
        self.w = L / self.n
        self.k = TWO_PI / L * torch.fft.rfftfreq(self.n, d=1.0 / self.n, dtype=x.dtype, device=x.device)

    def derivs(self, u):
        uh = torch.fft.rfft(u, dim=-1)
        ux = torch.fft.irfft(1j * self.k * uh, n=self.n, dim=-1)
        uxx = torch.fft.irfft(-(self.k ** 2) * uh, n=self.n, dim=-1)
        return ux, uxx

    def ip(self, Phi, F):                                  # Phi^T W F, batched: (B,n,G),(B,n,D) -> (B,G,D)
        return self.w * Phi.transpose(1, 2) @ F


# ============================================================================= basis
class BumpBasis(nn.Module):
    """G periodic bumps. Per bump: angle theta (centre = theta L / 2pi) and eta = tanh(sigma) in (-1, 1) setting
    the width between w_min and w_max. The model integrates sigma, so widths stay valid under ANY time step.
    kind = 'gauss': exp(-dist^2 / 2w^2);  kind = 'pou': softmax of those logits over bumps (partition of unity)."""

    def __init__(self, G, L, w_min, w_max, kind):
        super().__init__()
        self.G, self.L, self.w_min, self.w_max, self.kind = G, L, w_min, w_max, kind

    def width(self, eta):
        return self.w_min + (self.w_max - self.w_min) * (eta + 1) / 2

    def centre(self, theta):
        return torch.remainder(theta * self.L / TWO_PI, self.L)

    def _logits(self, x, theta, eta):
        mu, w = self.centre(theta), self.width(eta)                                  # (B, G)
        xx = x.unsqueeze(0) if x.dim() == 1 else x                                   # (1 or B, n)
        dist = torch.remainder(xx[:, :, None] - mu[:, None, :] + self.L / 2, self.L) - self.L / 2
        return -dist ** 2 / (2 * w[:, None, :] ** 2), dist, w

    def forward(self, x, theta, eta, theta_dot=None, eta_dot=None):
        ell, dist, w = self._logits(x, theta, eta)
        Phi = torch.exp(ell) if self.kind == "gauss" else torch.softmax(ell, dim=-1)
        if theta_dot is None:
            return Phi
        mu_dot = theta_dot * self.L / TWO_PI
        w_dot = (self.w_max - self.w_min) / 2 * eta_dot
        ell_dot = dist / w[:, None, :] ** 2 * mu_dot[:, None, :] + dist ** 2 / w[:, None, :] ** 3 * w_dot[:, None, :]
        if self.kind == "gauss":
            Phi_dot = Phi * ell_dot
        else:
            Phi_dot = Phi * (ell_dot - (Phi * ell_dot).sum(-1, keepdim=True))
        return Phi, Phi_dot


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
    """Linear interpolation of u (B, n) at positions pos (B, G) on the periodic grid."""
    idx = torch.remainder(pos, grid.L) / grid.L * grid.n
    i0 = torch.floor(idx).long() % grid.n
    frac = idx - torch.floor(idx)
    return (1 - frac) * torch.gather(u, 1, i0) + frac * torch.gather(u, 1, (i0 + 1) % grid.n)


# ============================================================================= small G x G linear algebra
SOLVE64 = True     # mass matrices of overlapping bumps can be ill-conditioned: solve the G x G systems in float64


def solve(A, B):
    """Solve A X = B (batched). float64 by default; if a matrix is numerically singular, retry with a tiny
    diagonal jitter instead of crashing. Non-finite inputs give non-finite outputs, caught by the training loop."""
    dtype = A.dtype
    if SOLVE64 and dtype != torch.float64:
        A, B = A.double(), B.double()
    X, info = torch.linalg.solve_ex(A, B)
    if (info != 0).any():
        scale = A.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1e-30)
        A = A + 1e-8 * scale[..., None, None] * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        X = torch.linalg.solve_ex(A, B)[0]
    return X.to(dtype)


# ============================================================================= FA correspondence and stability
def fa_correspondence(Qt, Kt, M, lam):
    """C = Q~ K~^T (K~ K~^T + lam M)^{-1}. The lam*M ridge makes C transform as A^T C A^{-T} under Phi -> Phi A."""
    S = Kt @ Kt.transpose(1, 2) + lam * M
    return solve(S, (Qt @ Kt.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)   # S symmetric


def _chol(M):
    Lc, info = torch.linalg.cholesky_ex(M)
    if (info != 0).any():
        scale = M.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
        Lc = torch.linalg.cholesky_ex(M + 1e-8 * scale[..., None, None] *
                                      torch.eye(M.shape[-1], dtype=M.dtype, device=M.device))[0]
    return Lc


def _to_chol(C, Lc):          # K = L^{-1} C L: the generator in field-orthonormal coordinates
    return torch.linalg.solve_triangular(Lc, C @ Lc, upper=False)


def _from_chol(K, Lc):        # C = L K L^{-1}
    return Lc @ torch.linalg.solve_triangular(Lc.transpose(1, 2), K.transpose(1, 2), upper=True).transpose(1, 2)


def cap_rate(C, M, kappa):
    """Training safeguard: scale the generator so ||L^{-1} C L||_F <= kappa. Then dE/dt <= 2 kappa E, so energy
    grows at most like exp(2 kappa t): no overflow. Smooth (no eigendecomposition) and frame covariant, since
    equivalent frames differ by an orthogonal map in these coordinates, which preserves the Frobenius norm."""
    if SOLVE64 and M.dtype != torch.float64:
        return cap_rate(C.double(), M.double(), kappa).to(C.dtype)
    Lc = _chol(M)
    K = _to_chol(C, Lc)
    nrm = torch.linalg.matrix_norm(K).clamp_min(1e-12)
    return _from_chol(K * torch.clamp(kappa / nrm, max=1.0)[:, None, None], Lc)


def stabilise(C, M):
    """Remove the positive part of sym(L^{-1} C L), L = chol(M): energy E = 1/2 d^T M^{-1} d then cannot grow
    through the physics term. Done in Cholesky (field-orthonormal) coordinates, so it is frame covariant."""
    if SOLVE64 and M.dtype != torch.float64:
        return stabilise(C.double(), M.double()).to(C.dtype)
    Lc = _chol(M)
    K = _to_chol(C, Lc)
    S = 0.5 * (K + K.transpose(1, 2))
    ev, U = torch.linalg.eigh(S)
    K = K - U @ torch.diag_embed(F.relu(ev)) @ U.transpose(1, 2)
    return _from_chol(K, Lc)


def psi_lower(X):
    """Lower triangle with halved diagonal. If M = L L^T then L^{-1} L_dot = psi_lower(L^{-1} M_dot L^{-T})."""
    return torch.tril(X, -1) + 0.5 * torch.diag_embed(torch.diagonal(X, dim1=-2, dim2=-1))


def transport_generator(Lc, H, transport=True):
    """Generator of the basis motion in Cholesky coordinates z = L^{-1} d:
        N = L^{-1} H^T L^{-T} - L^{-1} L_dot,   skew-symmetric (energy-neutral).
    With transport=False the first term is dropped, matching the d-form ablation d_dot = C d."""
    Li_HT = torch.linalg.solve_triangular(Lc, H.transpose(1, 2), upper=False)              # L^{-1} H^T
    A1 = torch.linalg.solve_triangular(Lc, Li_HT.transpose(1, 2), upper=False).transpose(1, 2)   # L^{-1} H^T L^{-T}
    X = A1 + A1.transpose(1, 2)                                                            # L^{-1} M_dot L^{-T}
    return (A1 if transport else 0.0) - psi_lower(X)


def cayley_apply(A, g, z, h):
    """Implicit midpoint for z' = A z + g with A, g frozen: (I - h/2 A) z_new = (I + h/2 A) z + h g.
    ||z_new||^2 - ||z||^2 = 2h zbar^T sym(A) zbar when g = 0, so a skew A preserves energy for ANY h."""
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    rhs = ((I + h / 2 * A) @ z.unsqueeze(-1)).squeeze(-1) + (h * g if g is not None else 0.0)
    return solve(I - h / 2 * A, rhs.unsqueeze(-1)).squeeze(-1)


# ============================================================================= model
class EvolvingFA(nn.Module):
    def __init__(self, cfg, grid, scales):
        super().__init__()
        self.cfg, self.grid = cfg, grid
        self.u_s, self.ux_s, self.uxx_s = scales
        L = grid.L
        self.basis = BumpBasis(cfg.G, L, cfg.w_min_frac * L, cfg.w_max_frac * L, cfg.basis)
        self.moving = cfg.model != "fixed"
        self.head = "field" if cfg.model == "field_head" else "fa"
        self.transport = cfg.model != "no_transport"

        H, D = cfg.hidden, cfg.desc
        self.enc = mlp([5, H, H, cfg.G])                                  # encoder: FA slicing over points
        self.feat = mlp([5, H, H, H])                                     # point features of the decoded field
        self.to_q, self.to_k = nn.Linear(H, D), nn.Linear(H, D)
        self.field_out = nn.Linear(H, 1)
        self.log_lam = nn.Parameter(torch.tensor(math.log(cfg.lam)))
        # residual placement: summarise the initial field near each grid bump (FA-style projection of point
        # features onto the bump) and map it to a bounded offset. Zero-initialised last layer: training starts
        # exactly on the even grid, but gradients flow from the first step (no gates, no angles, no 1/R terms).
        self.enc_feat = mlp([5, H, H])
        self.enc_out = mlp([2 * H, H, 2], last_zero=True)
        self.fs = mlp([7, H, H], act=nn.GELU)
        self.fs_out = mlp([2 * H, H, 2], last_zero=True)                  # starts with a still basis
        self.vel_scale = cfg.vel_scale
        self.sig_rate = cfg.sig_rate

    # ----- helpers
    def point_inputs(self, u, ux, uxx):
        ang = TWO_PI * self.grid.x / self.grid.L
        B = u.shape[0]
        return torch.stack([u / self.u_s, ux / self.ux_s, uxx / self.uxx_s,
                            torch.sin(ang).expand(B, -1), torch.cos(ang).expand(B, -1)], -1)

    def mass(self, Phi):
        M = self.grid.ip(Phi, Phi)
        eps = self.cfg.eps_M * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] + 1e-12
        return M + eps * torch.eye(M.shape[-1], device=M.device)

    # ----- encode u0 -> (theta, sigma, d);  widths use eta = tanh(sigma), so they stay in bounds for ANY sigma
    def encode(self, u0):
        B, G = u0.shape[0], self.cfg.G
        theta_u = (TWO_PI * (torch.arange(G, device=u0.device) + 0.5) / G).expand(B, -1)        # even grid
        eta_u = torch.full((B, G), 2 * (0.6 * self.grid.L / G - self.basis.w_min) /
                           (self.basis.w_max - self.basis.w_min) - 1, device=u0.device).clamp(-0.95, 0.95)
        if self.cfg.basis_init == "uniform":
            theta, eta = theta_u.clone(), eta_u
        elif self.cfg.basis_init == "residual":
            ux, uxx = self.grid.derivs(u0)
            feats = self.enc_feat(self.point_inputs(u0, ux, uxx))                                # (B, n, H)
            Phi_u = self.basis(self.grid.x, theta_u, eta_u)                                      # (B, n, G)
            pooled = Phi_u.transpose(1, 2) @ feats / Phi_u.sum(1).clamp_min(1e-8).unsqueeze(-1)  # (B, G, H)
            out = self.enc_out(torch.cat([pooled, pooled.mean(1, keepdim=True).expand_as(pooled)], -1))
            theta = theta_u + 0.45 * (TWO_PI / G) * torch.tanh(out[..., 0])   # < half a grid spacing: no crossings
            sig_u = torch.atanh(eta_u)
            eta = torch.tanh(sig_u + 1.5 * torch.tanh(out[..., 1])).clamp(-0.98, 0.98)
        else:
            ux, uxx = self.grid.derivs(u0)
            wts = torch.softmax(self.enc(self.point_inputs(u0, ux, uxx)) / self.cfg.enc_tau, dim=1)   # (B,n,G)
            ang = TWO_PI * self.grid.x / self.grid.L
            Rc = (wts * torch.cos(ang)[None, :, None]).sum(1)
            Rs = (wts * torch.sin(ang)[None, :, None]).sum(1)
            theta = torch.atan2(Rs, Rc)
            R = torch.sqrt(Rc ** 2 + Rs ** 2).clamp(1e-4, 1 - 1e-6)
            width = torch.sqrt(-2 * torch.log(R)) * self.grid.L / TWO_PI                         # circular std
            eta = (2 * (width - self.basis.w_min) / (self.basis.w_max - self.basis.w_min) - 1).clamp(-0.98, 0.98)
        Phi = self.basis(self.grid.x, theta, eta)
        d = self.grid.ip(Phi, u0.unsqueeze(-1)).squeeze(-1)             # field-metric fit: c0 = M^{-1} Phi^T W u0
        return theta, torch.atanh(eta), d

    def decode(self, theta, sig, d, x=None):
        eta = torch.tanh(sig)
        Phi = self.basis(self.grid.x, theta, eta)
        M = self.mass(Phi)
        c = solve(M, d.unsqueeze(-1)).squeeze(-1)
        Phi_out = Phi if x is None else self.basis(x, theta, eta)
        return (Phi_out @ c.unsqueeze(-1)).squeeze(-1), M, c, Phi

    # ----- right-hand side
    def rhs(self, theta, sig, d, aux=False):
        eta = torch.tanh(sig)
        u, M, c, Phi = self.decode(theta, sig, d)
        ux, uxx = self.grid.derivs(u)
        feats = self.feat(self.point_inputs(u, ux, uxx))                  # (B, n, H)

        if self.moving:
            pos = self.basis.centre(theta)
            loc = torch.stack([torch.cos(theta), torch.sin(theta), eta, c / self.u_s,
                               interp_periodic(u, pos, self.grid) / self.u_s,
                               interp_periodic(ux, pos, self.grid) / self.ux_s,
                               interp_periodic(uxx, pos, self.grid) / self.uxx_s], -1)       # (B, G, 7)
            h = self.fs(loc)
            out = self.fs_out(torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], -1))
            theta_dot = self.vel_scale * torch.tanh(out[..., 0])           # bounded basis velocity
            sig_dot = self.sig_rate * torch.tanh(out[..., 1])
            eta_dot = (1 - eta ** 2) * sig_dot                             # chain rule through eta = tanh(sigma)
            w = self.basis.width(eta)
            w_dot = (self.basis.w_max - self.basis.w_min) / 2 * eta_dot
            self._rate = ((theta_dot.abs() * self.grid.L / TWO_PI + w_dot.abs()) / w).amax().detach()
            _, Phi_dot = self.basis(self.grid.x, theta, eta, theta_dot, eta_dot)
            Hm = self.grid.ip(Phi, Phi_dot)
        else:
            theta_dot, sig_dot = torch.zeros_like(theta), torch.zeros_like(sig)
            Phi_dot = torch.zeros_like(Phi)
            self._rate = torch.zeros((), device=d.device)
            Hm = torch.zeros_like(M)

        lam = torch.exp(self.log_lam)
        C = None
        if self.head == "fa":
            Qt = self.grid.ip(Phi, self.to_q(feats))
            Kt = self.grid.ip(Phi, self.to_k(feats))
            C = fa_correspondence(Qt, Kt, M, lam)
            if self.cfg.stab:
                C = stabilise(C, M)
            if self.cfg.rate_cap > 0:
                C = cap_rate(C, M, self.cfg.rate_cap)
            phys = (C @ d.unsqueeze(-1)).squeeze(-1)
        else:
            phys = self.grid.ip(Phi, self.field_out(feats)).squeeze(-1) * self.u_s

        transport = (Hm.transpose(1, 2) @ c.unsqueeze(-1)).squeeze(-1)   # H^T M^{-1} d = H^T c
        d_dot = phys + (transport if self.transport else 0.0)
        if aux:
            return (theta_dot, sig_dot, d_dot), dict(Phi=Phi, Phi_dot=Phi_dot, M=M, H=Hm, c=c, u=u, phys=phys, C=C)
        return theta_dot, sig_dot, d_dot

    # ----- Cholesky-coordinate (z) formulation, integrated with the energy-stable midpoint/Cayley scheme
    def _hi(self, t):
        return t.double() if (SOLVE64 and t.dtype != torch.float64) else t

    def generator(self, theta, sig, z, aux=False):
        """Returns theta_dot, sig_dot, A, g with z' = A z + g in Cholesky coordinates z = L^{-1} d."""
        eta = torch.tanh(sig)
        Phi = self.basis(self.grid.x, theta, eta)
        M = self.mass(Phi)
        Lc = _chol(self._hi(M))
        c = torch.linalg.solve_triangular(Lc.transpose(1, 2), self._hi(z).unsqueeze(-1), upper=True)
        c = c.squeeze(-1).to(z.dtype)                                    # c = L^{-T} z
        u = (Phi @ c.unsqueeze(-1)).squeeze(-1)
        ux, uxx = self.grid.derivs(u)
        feats = self.feat(self.point_inputs(u, ux, uxx))

        if self.moving:
            pos = self.basis.centre(theta)
            loc = torch.stack([torch.cos(theta), torch.sin(theta), eta, c / self.u_s,
                               interp_periodic(u, pos, self.grid) / self.u_s,
                               interp_periodic(ux, pos, self.grid) / self.ux_s,
                               interp_periodic(uxx, pos, self.grid) / self.uxx_s], -1)
            h = self.fs(loc)
            out = self.fs_out(torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], -1))
            theta_dot = self.vel_scale * torch.tanh(out[..., 0])
            sig_dot = self.sig_rate * torch.tanh(out[..., 1])
            eta_dot = (1 - eta ** 2) * sig_dot
            w = self.basis.width(eta)
            w_dot = (self.basis.w_max - self.basis.w_min) / 2 * eta_dot
            self._rate = ((theta_dot.abs() * self.grid.L / TWO_PI + w_dot.abs()) / w).amax().detach()
            _, Phi_dot = self.basis(self.grid.x, theta, eta, theta_dot, eta_dot)
            A = transport_generator(Lc, self._hi(self.grid.ip(Phi, Phi_dot)), self.transport)
        else:
            theta_dot, sig_dot = torch.zeros_like(theta), torch.zeros_like(sig)
            self._rate = torch.zeros((), device=z.device)
            A = torch.zeros_like(Lc)

        g = None
        if self.head == "fa":
            Qt = self.grid.ip(Phi, self.to_q(feats))
            Kt = self.grid.ip(Phi, self.to_k(feats))
            C = fa_correspondence(Qt, Kt, M, torch.exp(self.log_lam))
            K = _to_chol(self._hi(C), Lc)                                  # physics generator, z coordinates
            if self.cfg.stab:
                S = 0.5 * (K + K.transpose(1, 2))
                ev, U = torch.linalg.eigh(S)
                K = K - U @ torch.diag_embed(F.relu(ev)) @ U.transpose(1, 2)
            if self.cfg.rate_cap > 0:
                nrm = torch.linalg.matrix_norm(K).clamp_min(1e-12)
                K = K * torch.clamp(self.cfg.rate_cap / nrm, max=1.0)[:, None, None]
            if getattr(self.cfg, "physics_explicit", False):
                g = (K @ self._hi(z).unsqueeze(-1)).squeeze(-1)
            else:
                A = A + K
        else:
            f = self.grid.ip(Phi, self.field_out(feats)).squeeze(-1) * self.u_s       # Phi^T W F(u): d-coordinates
            g = torch.linalg.solve_triangular(Lc, self._hi(f).unsqueeze(-1), upper=False).squeeze(-1)
        if aux:
            return theta_dot, sig_dot, A, g, dict(Lc=Lc, c=c, u=u, Phi=Phi, M=M)
        return theta_dot, sig_dot, A, g

    def step_cayley(self, state, h):
        """Explicit midpoint for the basis state, implicit midpoint (Cayley) for z with the generator frozen at
        the midpoint. Two generator evaluations per step; second order; energy-stable for any h."""
        th, sg, z = state
        thd, sgd, A, g = self.generator(th, sg, z)
        z_mid = cayley_apply(A, g, self._hi(z), h / 2).to(z.dtype)
        thd_m, sgd_m, A_m, g_m = self.generator(th + h / 2 * thd, sg + h / 2 * sgd, z_mid)
        z_new = cayley_apply(A_m, g_m, self._hi(z), h).to(z.dtype)
        return th + h * thd_m, sg + h * sgd_m, z_new

    def to_z(self, theta, sig, d):
        Lc = _chol(self._hi(self.mass(self.basis(self.grid.x, theta, torch.tanh(sig)))))
        return torch.linalg.solve_triangular(Lc, self._hi(d).unsqueeze(-1), upper=False).squeeze(-1).to(d.dtype)

    def decode_state(self, state, x=None):
        """Decoded field, mass matrix, coefficients, basis, energy for either state form."""
        th, sg, v = state
        eta = torch.tanh(sg)
        Phi = self.basis(self.grid.x, th, eta)
        M = self.mass(Phi)
        if self.cfg.integrator == "cayley":
            Lc = _chol(self._hi(M))
            c = torch.linalg.solve_triangular(Lc.transpose(1, 2), self._hi(v).unsqueeze(-1), upper=True)
            c = c.squeeze(-1).to(v.dtype)
            energy = 0.5 * (v * v).sum(-1)
        else:
            c = solve(M, v.unsqueeze(-1)).squeeze(-1)
            energy = 0.5 * (c * v).sum(-1)
        Phi_out = Phi if x is None else self.basis(x, th, eta)
        return (Phi_out @ c.unsqueeze(-1)).squeeze(-1), M, c, Phi, energy

    def rk4(self, state, h):
        k1 = self.rhs(*state)
        k2 = self.rhs(*[s + h / 2 * k for s, k in zip(state, k1)])
        k3 = self.rhs(*[s + h / 2 * k for s, k in zip(state, k2)])
        k4 = self.rhs(*[s + h * k for s, k in zip(state, k3)])
        return tuple(s + h / 6 * (a + 2 * b + 2 * cc + e) for s, a, b, cc, e in zip(state, k1, k2, k3, k4))

    def rollout(self, u0, t_query, dt, keep=False):
        """Integrate from t=0 to each (sorted) query time; returns decoded fields (B, K, n) and diagnostics.
        Step size is adapted CFL-style: RK4 is unstable for oscillation rates above ~2.8/h, and moving bumps
        induce rates ~ (bump speed / bump width), so each interval is sub-stepped to keep h * rate <= cfl."""
        state = self.encode(u0)
        cayley = self.cfg.integrator == "cayley"
        if cayley:
            state = (state[0], state[1], self.to_z(*state))
        with torch.no_grad():                                               # sets self._rate at t=0
            self.generator(*state) if cayley else self.rhs(*state)
        t_prev, outs, Ms, states, evals = 0.0, [], [], [], 1
        cfl, cap = getattr(self.cfg, "cfl", 1.0), getattr(self.cfg, "max_sub", 8)
        for tq in t_query:
            span = float(tq) - t_prev
            base = max(1, math.ceil(span / dt - 1e-9)) if span > 0 else 0
            rate = float(self._rate) + max(getattr(self.cfg, "rate_cap", 0.0), 0.0)
            n_sub = min(max(base, math.ceil(span * rate / cfl - 1e-9)), cap * base) if span > 0 else 0
            for _ in range(n_sub):
                state = self.step_cayley(state, span / n_sub) if cayley else self.rk4(state, span / n_sub)
                evals += 2 if cayley else 4
            u, M, c, _, _ = self.decode_state(state)
            outs.append(u)
            Ms.append(M)
            if keep:
                states.append(tuple(s.detach() for s in state))
            t_prev = float(tq)
        return torch.stack(outs, 1), torch.stack(Ms, 1), states, evals


def estimate_max_speed(u, t, L, T_train, n_traj=64, n_win=8):
    """How fast structures move in the training data: windowed cross-correlation of snapshots ~T_train/6 apart,
    95th percentile of local shift / time over windows that contain structure. Used to bound basis speeds."""
    u, t = u[:n_traj].cpu().numpy().astype(np.float64), t.cpu().numpy()
    n = u.shape[-1]
    dx, x = L / n, np.arange(n) * L / n
    lag = max(1, int(round((T_train / 6) / (t[1] - t[0]))))
    speeds = []
    for k in range(0, int(np.searchsorted(t, T_train)) - lag, lag):
        a, b = u[:, k] - u[:, k].mean(-1, keepdims=True), u[:, k + lag] - u[:, k + lag].mean(-1, keepdims=True)
        for c in np.arange(n_win) * L / n_win:
            win = np.exp(-(((x - c + L / 2) % L - L / 2) ** 2) / (2 * (L / n_win) ** 2))
            aw, bw = a * win, b * win
            energy = (aw ** 2).sum(-1)
            s = np.argmax(np.fft.irfft(np.fft.rfft(bw) * np.conj(np.fft.rfft(aw)), n=n), -1)
            s = np.where(s > n // 2, s - n, s)
            speeds.append((np.abs(s) * dx / (t[k + lag] - t[k]))[energy > 0.05 * energy.max()])
    return float(np.percentile(np.concatenate(speeds), 95))


# ============================================================================= losses and diagnostics
def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true, dim=-1) / (torch.linalg.norm(true, dim=-1) + 1e-8))


def nondegeneracy_penalty(Ms, floor):
    """Encourage (not enforce) M >= m I: penalise log of the smallest eigenvalue of the normalised mass matrix."""
    Ms = Ms.double()
    dg = Ms.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
    Mn = Ms / torch.sqrt(dg[..., :, None] * dg[..., None, :])
    lmin = torch.linalg.eigvalsh(Mn)[..., 0].clamp_min(1e-12)
    return F.relu(math.log(floor) - torch.log(lmin)).mean().float()


@torch.no_grad()
def projection_error(Phi, u, w):
    """Best relative L2 error of u in span(Phi) under the W inner product (how good the basis could be)."""
    M = w * Phi.transpose(1, 2) @ Phi
    M = M + 1e-10 * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(M.shape[-1], device=M.device)
    c = solve(M, w * Phi.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)
    return rel_l2((Phi @ c.unsqueeze(-1)).squeeze(-1), u)


@torch.no_grad()
def max_subspace_angle(Phi1, Phi2):
    """Largest principal angle (radians) between two spans, W uniform so plain QR suffices."""
    Q1, _ = torch.linalg.qr(Phi1)
    Q2, _ = torch.linalg.qr(Phi2)
    s = torch.linalg.svdvals(Q1.transpose(1, 2) @ Q2).clamp(-1, 1)
    return torch.arccos(s.min(-1).values)


# ============================================================================= tests
def run_tests():
    torch.manual_seed(0)
    dt64, ok = torch.float64, True

    def check(name, err, tol):
        nonlocal ok
        good = err < tol
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name:64s} {err:.2e}")

    L, n, G = TWO_PI, 128, 5
    x = torch.arange(n, dtype=dt64) * L / n
    grid = Grid(x, L)

    # 1. analytic basis derivative
    for kind in ("gauss", "pou"):
        basis = BumpBasis(G, L, 0.05, 1.5, kind)
        th, et = torch.rand(2, G, dtype=dt64) * TWO_PI, torch.rand(2, G, dtype=dt64) * 1.6 - 0.8
        thd, etd = torch.randn(2, G, dtype=dt64), torch.randn(2, G, dtype=dt64)
        h = 1e-6
        fd = (basis(x, th + h * thd, et + h * etd) - basis(x, th - h * thd, et - h * etd)) / (2 * h)
        _, an = basis(x, th, et, thd, etd)
        check(f"basis '{kind}': analytic Phi_dot vs finite difference", ((an - fd).abs().max() / fd.abs().max()).item(), 1e-7)

    # 2. FA correspondence transforms covariantly with lam*M ridge (and not with lam*I)
    B, D = 2, 16
    Phi = torch.randn(B, n, G, dtype=dt64)
    FQ, FK = torch.randn(B, n, D, dtype=dt64), torch.randn(B, n, D, dtype=dt64)
    A = torch.eye(G, dtype=dt64) + 0.4 * torch.randn(B, G, G, dtype=dt64)
    M, Mt = grid.ip(Phi, Phi), grid.ip(Phi @ A, Phi @ A)
    C = fa_correspondence(grid.ip(Phi, FQ), grid.ip(Phi, FK), M, 0.3)
    Ct = fa_correspondence(grid.ip(Phi @ A, FQ), grid.ip(Phi @ A, FK), Mt, 0.3)
    check("FA solve: C~ = A^T C A^-T with lam*M ridge",
          (Ct - A.transpose(1, 2) @ C @ torch.linalg.inv(A.transpose(1, 2))).abs().max().item(), 1e-9)
    Cs, Cst = stabilise(C, M), stabilise(Ct, Mt)
    check("stability projection: frame covariant",
          (Cst - A.transpose(1, 2) @ Cs @ torch.linalg.inv(A.transpose(1, 2))).abs().max().item(), 1e-9)
    Cc, Cct = cap_rate(C, M, 0.5), cap_rate(Ct, Mt, 0.5)
    check("rate cap: frame covariant",
          (Cct - A.transpose(1, 2) @ Cc @ torch.linalg.inv(A.transpose(1, 2))).abs().max().item(), 1e-9)
    Lc = torch.linalg.cholesky(M)
    check("rate cap: ||L^-1 C L||_F <= kappa",
          max((torch.linalg.matrix_norm(_to_chol(Cc, Lc)) - 0.5).max().item(), 0.0), 1e-10)
    K = torch.linalg.solve_triangular(Lc, Cs @ Lc, upper=False)
    check("stability projection: max eig of sym(L^-1 C L) <= 0",
          max(torch.linalg.eigvalsh(0.5 * (K + K.transpose(1, 2))).max().item(), 0.0), 1e-10)

    # 3. model identities on a random state: energy identity and velocity decomposition
    class Cfg:
        G = 5; basis = "gauss"; w_min_frac = 0.01; w_max_frac = 0.25; model = "full"; hidden = 32; desc = 16
        lam = 0.1; vel_scale = 1.0; sig_rate = 1.0; eps_M = 0.0; basis_init = "encoder"; enc_tau = 1.0; stab = False
        rate_cap = 0.0; cfl = 1.0; max_sub = 8; integrator = "rk4"; physics_explicit = False
    for stab in (False, True):
        cfg = Cfg(); cfg.stab = stab
        model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0)).double()
        with torch.no_grad():
            for p in model.fs_out.parameters():
                p.normal_(0, 0.3)                        # make the basis actually move for the test
        theta = torch.rand(B, G, dtype=dt64) * TWO_PI
        sig = torch.atanh(torch.rand(B, G, dtype=dt64) * 1.2 - 0.6)
        d = torch.randn(B, G, dtype=dt64)
        with torch.no_grad():
            (thd, sgd, dd), a = model.rhs(theta, sig, d, aux=True)
        c, Mm, Hm = a["c"], a["M"], a["H"]
        cond = torch.linalg.cond(Mm).max().item()
        tol = max(1e-9, 1e-12 * cond)                                   # round-off grows like eps * cond(M)
        Mdot = Hm + Hm.transpose(1, 2)
        t1, t2 = (c * dd).sum(-1), 0.5 * (c.unsqueeze(1) @ Mdot @ c.unsqueeze(-1)).squeeze()
        E_dot, target = t1 - t2, (c * a["phys"]).sum(-1)
        scale = t1.abs() + t2.abs() + target.abs() + 1e-300          # size of the terms being cancelled
        check(f"energy identity E' = c^T phys, relative (stab={stab}, cond M {cond:.0e})",
              ((E_dot - target).abs() / scale).max().item(), tol)
        if stab:
            check("with projection: E' <= 0 (relative)", max((E_dot / scale).max().item(), 0.0), tol)
        c_dot = torch.linalg.solve(Mm, (dd - (Mdot @ c.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        u_dot = (a["Phi_dot"] @ c.unsqueeze(-1) + a["Phi"] @ c_dot.unsqueeze(-1)).squeeze(-1)
        P_out = a["Phi_dot"] @ c.unsqueeze(-1) - a["Phi"] @ torch.linalg.solve(Mm, Hm @ c.unsqueeze(-1))
        in_span = a["Phi"] @ torch.linalg.solve(Mm, a["phys"].unsqueeze(-1))
        check(f"velocity decomposition (I-P)Phi_dot c + Phi M^-1 C d (stab={stab})",
              (u_dot - (P_out + in_span).squeeze(-1)).abs().max().item(), 1e-9)
        ip = lambda f, g: grid.w * (f * g).sum((1, 2))
        cosine = ip(P_out, in_span) / torch.sqrt(ip(P_out, P_out) * ip(in_span, in_span))
        check(f"decomposition parts are W-orthogonal, cosine (stab={stab})", cosine.abs().max().item(), tol)
        with torch.no_grad():                            # any sigma, however extreme, gives a valid basis
            extreme = torch.tensor([[-40.0, -5.0, 0.0, 5.0, 40.0]], dtype=dt64).expand(B, -1)
            w = model.basis.width(torch.tanh(extreme))
            Mx = model.mass(model.basis(model.grid.x, theta, torch.tanh(extreme)))
            ok_w = bool(((w >= model.basis.w_min) & (w <= model.basis.w_max)).all()) and \
                bool(torch.isfinite(Mx).all()) and bool((torch.linalg.eigvalsh(Mx)[:, 0] > 0).all())
        check(f"extreme sigma: widths in bounds, mass matrix finite and positive definite (stab={stab})",
              0.0 if ok_w else 1.0, 0.5)

    # 4. Cayley integrator: transport generator is skew; energy never increases with --stab, even for large steps;
    #    and the z-form agrees with the d-form (RK4) for small steps
    Phi = torch.randn(B, n, G, dtype=dt64); Phid = torch.randn(B, n, G, dtype=dt64)
    Lc = torch.linalg.cholesky(grid.ip(Phi, Phi))
    Nz = transport_generator(Lc, grid.ip(Phi, Phid))
    check("Cayley: transport generator N is skew (relative)",
          ((Nz + Nz.transpose(1, 2)).abs().max() / Nz.abs().max()).item(), 1e-12)
    cfg = Cfg(); cfg.stab = True; cfg.integrator = "cayley"
    model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0)).double()
    with torch.no_grad():
        for p_ in model.fs_out.parameters():
            p_.normal_(0, 0.3)
        st = (torch.rand(B, G, dtype=dt64) * TWO_PI, torch.atanh(torch.rand(B, G, dtype=dt64) * 1.2 - 0.6),
              torch.randn(B, G, dtype=dt64))
        energies = [0.5 * (st[2] ** 2).sum(-1)]
        for _ in range(10):
            st = model.step_cayley(st, 0.5)
            energies.append(0.5 * (st[2] ** 2).sum(-1))
        E = torch.stack(energies, 1)
        worst = ((E[:, 1:] - E[:, :-1]) / E[:, :-1]).max().item()
    check("Cayley + stab: energy non-increasing over 10 steps of h=0.5 (relative)", max(worst, 0.0), 1e-10)
    # consistency of the two formulations: the Cayley z-form must converge to the RK4 d-form solution at
    # second order as h shrinks (a formulation bug would make the error plateau instead)
    cfg = Cfg(); cfg.integrator = "cayley"; cfg.rate_cap = 5.0         # same capped physics in both forms
    model = EvolvingFA(cfg, grid, (1.0, 1.0, 1.0)).double()
    with torch.no_grad():
        for p_ in model.fs_out.parameters():
            p_.normal_(0, 0.1)
        th0 = (TWO_PI * (torch.arange(G, dtype=dt64) + 0.5) / G).expand(B, -1) + 0.1 * torch.randn(B, G, dtype=dt64)
        sg0 = torch.atanh(torch.rand(B, G, dtype=dt64) * 0.6 - 0.6)
        u_smooth = (torch.sin(x) + 0.5 * torch.cos(2 * x)).expand(B, -1)
        d0 = grid.ip(model.basis(x, th0, torch.tanh(sg0)), u_smooth.unsqueeze(-1)).squeeze(-1)
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
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    print(f"       Cayley vs fine RK4: errors {', '.join(f'{e:.1e}' for e in errs)}; "
          f"ratios {', '.join(f'{r:.2f}' for r in ratios)} (4 = second order)")
    check("Cayley z-form converges to RK4 d-form at second order (min ratio >= 3)", max(3.0 - min(ratios), 0.0), 1e-12)
    check("Cayley z-form matches RK4 d-form at the finest step (relative)", errs[-1], 1e-3)

    # 5. smoke test: every variant runs forward and backward in float32
    xs = torch.arange(64, dtype=torch.float32) * L / 64
    g32 = Grid(xs, L)
    u0 = torch.exp(-((xs - 2.0) ** 2) / 0.1).expand(2, -1).contiguous()
    for variant, integ in [(v, i) for i in ("cayley", "rk4") for v in ("full", "fixed", "no_transport", "field_head")]:
        cfg = Cfg(); cfg.model = variant; cfg.eps_M = 1e-6; cfg.basis_init = "residual"; cfg.rate_cap = 3.0
        cfg.integrator = integ
        m = EvolvingFA(cfg, g32, (0.5, 2.0, 10.0))
        pred, Ms, _, ev = m.rollout(u0, [0.1, 0.2], dt=0.05)
        loss = rel_l2(pred, u0.unsqueeze(1).expand_as(pred)).mean() + 0.01 * nondegeneracy_penalty(Ms, 1e-3)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        ok &= finite
        print(f"[{'PASS' if finite else 'FAIL'}] {variant:12s} {integ:6s} rollout + backward ({ev} rhs evals), "
              f"loss {loss.item():.3f}")
    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return ok


# ============================================================================= train / evaluate
def time_splits(t, T_train):
    idx = torch.arange(len(t))
    seen = [i for i in idx.tolist() if i > 0 and i % 2 == 0 and t[i] <= T_train + 1e-9]
    unseen = [i for i in idx.tolist() if i % 2 == 1 and t[i] <= T_train + 1e-9]
    extrap = [i for i in idx.tolist() if t[i] > T_train + 1e-9]
    return seen, unseen, extrap


@torch.no_grad()
def evaluate(model, data, cfg, splits, max_traj=None):
    model.eval()
    seen, unseen, extrap = splits
    allq = sorted(seen + unseen + extrap)
    tq = data.t[allq].tolist()
    errs, proj, angles, energy, centres, rt, evals_total = [], [], [], [], [], 0.0, 0
    N = data.u.shape[0] if max_traj is None else min(max_traj, data.u.shape[0])
    for i in range(0, N, cfg.batch_eval):
        u = data.u[i:i + cfg.batch_eval]
        if u.is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        pred, Ms, states, ev = model.rollout(u[:, 0], tq, cfg.dt, keep=True)
        if u.is_cuda:
            torch.cuda.synchronize()
        rt += time.time() - t0
        evals_total += ev
        true = u[:, allq]
        errs.append(rel_l2(pred, true))
        p_err, ang, en, cen, prev = [], [], [], [], None
        for k, st in enumerate(states):
            _, _, _, Phi, e = model.decode_state(st)
            p_err.append(projection_error(Phi, true[:, k], model.grid.w))
            ang.append(torch.zeros(u.shape[0], device=u.device) if prev is None else max_subspace_angle(prev, Phi))
            en.append(e)
            cen.append(model.basis.centre(st[0]))
            prev = Phi
        proj.append(torch.stack(p_err, 1)); angles.append(torch.stack(ang, 1))
        energy.append(torch.stack(en, 1)); centres.append(torch.stack(cen, 1))
    errs, proj = torch.cat(errs).cpu(), torch.cat(proj).cpu()
    pos = {q: j for j, q in enumerate(allq)}
    summary = {name: errs[:, [pos[q] for q in qs]].mean().item() if qs else float("nan")
               for name, qs in (("seen", seen), ("unseen", unseen), ("extrapolation", extrap))}
    summary.update(runtime_per_rollout_s=rt / max(1, math.ceil(N / cfg.batch_eval)),
                   rhs_evals_per_rollout=evals_total / max(1, math.ceil(N / cfg.batch_eval)))
    model.train()
    return dict(summary=summary, t=data.t[allq].tolist(),
                err_vs_t=errs.mean(0).tolist(), proj_err_vs_t=proj.mean(0).tolist(),
                subspace_angle_vs_t=torch.cat(angles).cpu().mean(0).tolist(),
                energy_example=torch.cat(energy)[0].cpu().tolist(),
                centres_example=torch.cat(centres)[0].cpu().tolist(),
                pred_example=None)


def plot(result, data, model, cfg, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t, T_train = np.array(result["t"]), cfg.T_train
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(t, result["err_vs_t"], label="forecast error")
    ax[0].plot(t, result["proj_err_vs_t"], "--", label="best error in current span")
    ax[0].axvline(T_train, c="gray", ls=":", label="end of training horizon")
    ax[0].set(yscale="log", xlabel="t", ylabel="relative L2", title=f"{cfg.problem} / {cfg.model} / G={cfg.G}")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(t, result["subspace_angle_vs_t"])
    ax[1].set(xlabel="t", ylabel="max principal angle (rad)", title="subspace motion between query times")
    ax[1].grid(alpha=0.3)
    with torch.no_grad():
        u = data.u[:1]
        pred, _, states, _ = model.rollout(u[:, 0], data.t[1:].tolist(), cfg.dt, keep=True)
    x = data.x.cpu().numpy()
    err = (pred[0] - u[0, 1:]).abs().cpu().numpy()
    ax[2].imshow(err, aspect="auto", origin="lower", cmap="magma",
                 extent=[x[0], x[-1], float(data.t[1]), float(data.t[-1])])
    cen = torch.stack([model.basis.centre(s[0])[0] for s in states]).cpu().numpy()
    for g in range(cen.shape[1]):
        ax[2].scatter(cen[:, g], data.t[1:].numpy(), s=1, c="cyan")
    ax[2].axhline(T_train, c="white", ls=":")
    ax[2].set(xlabel="x", ylabel="t", title="|error| on test trajectory 0, basis centres in cyan")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results.png"), dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", action="store_true")
    p.add_argument("--problem", choices=list(PROBLEM_DEFAULTS), default="kdv")
    p.add_argument("--model", choices=["full", "fixed", "no_transport", "field_head"], default="full")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--basis", choices=["gauss", "pou"], default=None)
    p.add_argument("--basis_init", choices=["residual", "encoder", "uniform"], default="residual",
                   help="residual: even grid plus bounded learned offsets from local summaries (default)")
    p.add_argument("--w_min_frac", type=float, default=None, help="narrowest bump, as a fraction of L")
    p.add_argument("--w_max_frac", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--desc", type=int, default=32, help="number of FA descriptor channels D")
    p.add_argument("--lam", type=float, default=0.1, help="initial FA ridge (learned)")
    p.add_argument("--stab", action="store_true", help="energy-stability projection")
    p.add_argument("--rate_cap", type=float, default=None, help="max ||L^-1 C L||_F (default 10/T_train; 0 = off)")
    p.add_argument("--eps_M", type=float, default=1e-6)
    p.add_argument("--solve32", action="store_true", help="do the G x G solves in float32 (faster, less robust)")
    p.add_argument("--enc_tau", type=float, default=1.0)
    p.add_argument("--vel_scale", type=float, default=None,
                   help="max basis angular speed in rad/time (default: 2x the speed estimated from the data)")
    p.add_argument("--cfl", type=float, default=1.0, help="step so that dt * (bump speed / bump width) <= cfl")
    p.add_argument("--integrator", choices=["cayley", "rk4"], default="cayley",
                   help="cayley: energy-stable midpoint in Cholesky coordinates (default); rk4: classical, d-form")
    p.add_argument("--physics_explicit", action="store_true",
                   help="treat the FA physics explicitly, as the generic head must be (fair ablation)")
    p.add_argument("--max_sub", type=int, default=8, help="cap on adaptive sub-stepping (multiple of the base)")
    p.add_argument("--sig_rate", type=float, default=None, help="max rate of the width variable (default 4/T_train)")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--stride", type=int, default=2, help="quadrature = every stride-th grid point")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--warm", type=int, default=1000, help="iterations over which the horizon grows")
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
    global SOLVE64
    SOLVE64 = not cfg.solve32

    defaults = PROBLEM_DEFAULTS[cfg.problem]
    cfg.basis = cfg.basis or defaults["basis"]
    cfg.w_min_frac = cfg.w_min_frac or defaults["w_min_frac"]
    cfg.dt = cfg.dt or defaults["dt"]
    cfg.out = cfg.out or f"runs/{cfg.problem}_{cfg.model}_G{cfg.G}"
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train = Trajectories(os.path.join(cfg.data, f"{cfg.problem}_train.npz"), cfg.stride, cfg.device)
    test = Trajectories(os.path.join(cfg.data, f"{cfg.problem}_test.npz"), cfg.stride, cfg.device)
    cfg.T_train = float(train.t[-1]) * defaults["train_frac"]
    speed = estimate_max_speed(train.u, train.t, train.L, cfg.T_train)
    cfg.vel_scale = cfg.vel_scale or 2.0 * speed * TWO_PI / train.L
    cfg.sig_rate = cfg.sig_rate or 4.0 / cfg.T_train
    cfg.rate_cap = 10.0 / cfg.T_train if cfg.rate_cap is None else cfg.rate_cap
    grid = Grid(train.x, train.L)
    with torch.no_grad():                                      # input normalisation from training data
        sample = train.u[:64].reshape(-1, grid.n)
        ux, uxx = grid.derivs(sample)
        scales = (sample.std().item() + 1e-6, ux.std().item() + 1e-6, uxx.std().item() + 1e-6)
    splits = time_splits(train.t, cfg.T_train)
    model = EvolvingFA(cfg, grid, scales).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iters)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"=== {cfg.problem} / {cfg.model} / basis {cfg.basis} / G={cfg.G} / {n_params:,} params / "
          f"T_train={cfg.T_train} / dt={cfg.dt} / quadrature n={grid.n}")
    print(f"    structure speed in data ~{speed:.3f} -> basis speed limit {cfg.vel_scale * train.L / TWO_PI:.3f}; "
          f"narrowest bump {model.basis.w_min:.3f}; rate cap {cfg.rate_cap:.3g}")

    seen_t = train.t[splits[0]]
    history, t0, run_loss, skipped, window = [], time.time(), 0.0, 0, []
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
                raise RuntimeError(f"{skipped} of {it} batches were non-finite: the dynamics are blowing up. "
                                   f"Try --stab, a smaller --dt, or a lower --lr.")
            continue
        (loss + cfg.reg_nondeg * reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += loss.item()
        window.append(loss.item())
        if it % cfg.log_every == 0:
            print(f"  it {it:5d}  horizon {T_cur:.2f}  train rel-L2 mean {np.mean(window) if window else float('nan'):.4f}"
                  f" median {np.median(window) if window else float('nan'):.4f}  nondeg {reg.item():.3f}  "
                  f"skipped {skipped}  "
                  f"({(time.time() - t0) / it:.2f} s/it, {time.time() - t0:.0f}s)")
            run_loss, window = 0.0, []
        if it % cfg.eval_every == 0 or it == cfg.iters:
            res = evaluate(model, test, cfg, splits, max_traj=cfg.max_eval_traj)
            s = res["summary"]
            print(f"  [eval] seen {s['seen']:.4f}  unseen {s['unseen']:.4f}  extrapolation {s['extrapolation']:.4f}  "
                  f"| {s['runtime_per_rollout_s']:.2f}s / rollout batch, {s['rhs_evals_per_rollout']:.0f} rhs evals")
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