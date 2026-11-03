"""
Moving-structure PDE problems for the evolving-basis FA prototype. Pure NumPy.

  advect   u_t + c u_x = nu u_xx on [0, 2pi), periodic. Initial condition: 1-3 narrow Gaussian
           pulses. Exact solution (each pulse translates at speed c and spreads). Linear transport:
           a fixed basis needs many modes, a moving one needs few.
  kdv      u_t + 6 u u_x + u_xxx = 0 on [0, L), periodic. One soliton u = (k/2) sech^2(sqrt(k)/2 (x - x0 - k t)):
           taller solitons are narrower AND faster, so the motion is a nonlinear function of the field.
           Exact on the line; periodic images make the error negligible for these widths.
  burgers  u_t + u u_x = nu u_xx on [0, 2pi), periodic. Random smooth initial condition; fronts steepen
           and travel. Solved with an integrating-factor RK4 pseudo-spectral method with 2/3 dealiasing,
           on a finer grid than the one saved.

Every dataset is saved as one .npz: u (N, T, X) float32, t (T,), x (X,), L, plus the problem's
parameters. Exact problems also store per-trajectory parameters so solutions can be evaluated at any
time and resolution (fa_problems.exact_solution).

Usage:
  python fa_problems.py --problem advect --out data/                # writes data/advect_{train,test}.npz
  python fa_problems.py --problem all --out data/ --preview          # all three plus a preview figure
"""
import argparse
import json
import os
import time

import numpy as np

CONFIGS = {
    # domain length, end time, save interval, grid, and problem constants
    "advect":  dict(L=2 * np.pi, T=4.0, dt_save=0.05, n=256, c=1.0, nu=0.002,
                    sigma=(0.08, 0.20), amp=(0.5, 1.5), pulses=(1, 3)),
    "kdv":     dict(L=40.0, T=6.0, dt_save=0.05, n=256, speed=(0.6, 2.4)),
    "burgers": dict(L=2 * np.pi, T=3.0, dt_save=0.025, n=256, nu=0.04, n_solve=512, dt_solve=2.5e-3,
                    modes=3, amp=0.8),
}


# ----------------------------------------------------------------------------- exact problems
def _wrap(z, L):
    return (z + L / 2) % L - L / 2


def advect_solution(params, x, t, L, c, nu):
    """params: dict with mu, sigma, amp of shape (N, P). x: (X,), t: (T,). Returns (N, T, X)."""
    mu, sig, amp = params["mu"], params["sigma"], params["amp"]
    tt = t[None, :, None, None]                                  # (1, T, 1, 1)
    s2 = sig[:, None, :, None] ** 2 + 2 * nu * tt                # spread variance (N, T, P, 1)
    centre = mu[:, None, :, None] + c * tt
    dist = _wrap(x[None, None, None, :] - centre, L)             # (N, T, P, X)
    shape = np.exp(-dist ** 2 / (2 * s2)) * (sig[:, None, :, None] / np.sqrt(s2))
    return (amp[:, None, :, None] * shape).sum(2)


def kdv_solution(params, x, t, L):
    """params: k (speed), x0 of shape (N,). Single soliton; distance wrapped onto the periodic domain."""
    k, x0 = params["k"][:, None, None], params["x0"][:, None, None]
    a = 0.5 * np.sqrt(k) * np.abs(_wrap(x[None, None, :] - x0 - k * t[None, :, None], L))
    e = np.exp(-2 * a)                                           # stable sech^2(a) = 4e^{-2a}/(1+e^{-2a})^2
    return 0.5 * k * 4 * e / (1 + e) ** 2


def exact_solution(problem, params, x, t, cfg=None):
    cfg = cfg or CONFIGS[problem]
    if problem == "advect":
        return advect_solution(params, x, t, cfg["L"], cfg["c"], cfg["nu"])
    if problem == "kdv":
        return kdv_solution(params, x, t, cfg["L"])
    raise ValueError(f"{problem} has no closed-form solution")


def sample_advect(N, rng, cfg):
    P = cfg["pulses"][1]
    n_p = rng.integers(cfg["pulses"][0], cfg["pulses"][1] + 1, size=N)
    active = (np.arange(P)[None, :] < n_p[:, None]).astype(float)
    return dict(mu=rng.uniform(0, cfg["L"], (N, P)),
                sigma=rng.uniform(*cfg["sigma"], (N, P)),
                amp=rng.uniform(*cfg["amp"], (N, P)) * active)


def sample_kdv(N, rng, cfg):
    return dict(k=rng.uniform(*cfg["speed"], N), x0=rng.uniform(0, cfg["L"], N))


# ----------------------------------------------------------------------------- Burgers solver
def solve_burgers(u0, L, nu, T, dt, save_every, stride=1):
    """Integrating-factor RK4 (Lawson) for u_t + (u^2/2)_x = nu u_xx. u0: (N, n). Returns (N, S, n)."""
    N, n = u0.shape
    k = np.fft.rfftfreq(n, d=L / (2 * np.pi * n))                # angular wavenumbers
    dealias = (np.abs(k) < (2 / 3) * k.max()).astype(float)
    lin = -nu * k ** 2

    def nonlin(vh):                                             # -(u^2/2)_x in Fourier space
        v = np.fft.irfft(vh * dealias, n=n)
        return -0.5j * k * np.fft.rfft(v * v) * dealias

    e_half, e_full = np.exp(lin * dt / 2), np.exp(lin * dt)
    uh = np.fft.rfft(u0)
    steps = int(round(T / dt))
    frames = [np.fft.irfft(uh, n=n)[:, ::stride].astype(np.float32)]
    for i in range(1, steps + 1):
        a = nonlin(uh)
        b = nonlin(e_half * (uh + dt / 2 * a))
        c = nonlin(e_half * uh + dt / 2 * b)
        d = nonlin(e_full * uh + dt * e_half * c)
        uh = e_full * uh + dt / 6 * (e_full * a + 2 * e_half * (b + c) + d)
        if i % save_every == 0:
            frames.append(np.fft.irfft(uh, n=n)[:, ::stride].astype(np.float32))
    return np.stack(frames, 1)


def sample_burgers(N, rng, cfg):
    n = cfg["n_solve"]
    x = np.arange(n) * cfg["L"] / n
    u0 = np.zeros((N, n))
    for m in range(1, cfg["modes"] + 1):
        a, b = rng.normal(size=(2, N, 1)) / m
        u0 += a * np.cos(m * x) + b * np.sin(m * x)
    u0 *= cfg["amp"] / (np.abs(u0).max(1, keepdims=True) + 1e-9)   # peak amplitude = amp
    return u0


# ----------------------------------------------------------------------------- driver
def generate(problem, N, seed, cfg=None):
    cfg = cfg or CONFIGS[problem]
    rng = np.random.default_rng(seed)
    x = np.arange(cfg["n"]) * cfg["L"] / cfg["n"]
    t = np.round(np.arange(0, cfg["T"] + 1e-9, cfg["dt_save"]), 10)
    if problem in ("advect", "kdv"):
        params = sample_advect(N, rng, cfg) if problem == "advect" else sample_kdv(N, rng, cfg)
        u = np.concatenate([exact_solution(problem, {k: v[i:i + 100] for k, v in params.items()}, x, t, cfg)
                            .astype(np.float32) for i in range(0, N, 100)])
    else:
        params = {}
        u0 = sample_burgers(N, rng, cfg)
        save_every = int(round(cfg["dt_save"] / cfg["dt_solve"]))
        assert abs(save_every * cfg["dt_solve"] - cfg["dt_save"]) < 1e-12, "dt_save must be a multiple of dt_solve"
        u = solve_burgers(u0, cfg["L"], cfg["nu"], cfg["T"], cfg["dt_solve"], save_every,
                          stride=cfg["n_solve"] // cfg["n"])
    assert u.shape[1] == len(t), f"time grid mismatch: {u.shape[1]} frames vs {len(t)} times"
    return dict(u=u.astype(np.float32), t=t, x=x, L=cfg["L"], problem=problem,
                config=json.dumps({k: v for k, v in cfg.items()}, default=float),
                **{f"param_{k}": v for k, v in params.items()})


def preview(datasets, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 6.5))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, (name, ds) in enumerate(datasets.items()):
        u, t, x = ds["u"], ds["t"], ds["x"]
        axes[0, j].imshow(u[0], aspect="auto", origin="lower", cmap="RdBu_r",
                          extent=[x[0], x[-1], t[0], t[-1]],
                          vmin=-np.abs(u[0]).max(), vmax=np.abs(u[0]).max())
        axes[0, j].set(title=f"{name}: trajectory 0", xlabel="x", ylabel="t")
        for i, ti in enumerate(np.linspace(0, len(t) - 1, 4).astype(int)):
            axes[1, j].plot(x, u[1, ti], label=f"t={t[ti]:.2f}", color=plt.cm.viridis(i / 3))
        axes[1, j].set(title=f"{name}: trajectory 1 snapshots", xlabel="x")
        axes[1, j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", choices=["advect", "kdv", "burgers", "all"], default="all")
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data")
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    problems = ["advect", "kdv", "burgers"] if a.problem == "all" else [a.problem]
    shown = {}
    for prob in problems:
        for split, N, seed in (("train", a.n_train, a.seed), ("test", a.n_test, a.seed + 1000)):
            t0 = time.time()
            ds = generate(prob, N, seed)
            path = os.path.join(a.out, f"{prob}_{split}.npz")
            np.savez_compressed(path, **ds)
            print(f"{prob:8s} {split:5s}: u {ds['u'].shape}, |u|max {np.abs(ds['u']).max():.2f}, "
                  f"{time.time() - t0:.1f}s -> {path}")
            if split == "test":
                shown[prob] = ds
    if a.preview:
        preview(shown, os.path.join(a.out, "preview.png"))


if __name__ == "__main__":
    main()
