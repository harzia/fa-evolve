"""
Compare runs of fa_evolve.py (v2). Reads every <runs>/<problem>_<variant>_G<G>_s<seed>/results.json.

Prints a markdown table (mean +- std over seeds) and saves <runs>/compare_<problem>.png:
  left    forecast error vs time per variant at one G
  middle  best error in the current span (is the basis doing useful work?)
  right   error vs number of basis functions G, per variant (in horizon and beyond it)

Usage:  python fa_compare.py --runs /content/drive/MyDrive/fa_runs_v2 --problem kdv --G 8
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ORDER = ["fixed", "no_transport", "nosolve", "field_head", "token_head", "full_explicit", "full"]
NAMES = {"fixed": "fixed basis + FA", "no_transport": "moving + FA, no transport",
         "nosolve": "moving + FA without the solve", "field_head": "moving + generic local head",
         "token_head": "moving + generic nonlocal head", "full_explicit": "full, physics explicit",
         "full": "full (moving + FA + transport)"}
PATTERN = r"(full_explicit|no_transport|field_head|token_head|nosolve|fixed|full)_G(\d+)(?:_s(\d+))?$"


def load(runs, problem):
    out = {}
    for f in glob.glob(os.path.join(runs, f"{problem}_*", "results.json")):
        m = re.match(rf"{problem}_{PATTERN}", os.path.basename(os.path.dirname(f)))
        if m:
            out.setdefault((m.group(1), int(m.group(2))), []).append(json.load(open(f)))
    return out


def stat(rs, key):
    v = np.array([r["summary"][key] for r in rs], float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), len(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--problem", default="kdv")
    p.add_argument("--G", type=int, default=None)
    a = p.parse_args()
    res = load(a.runs, a.problem)
    if not res:
        raise SystemExit(f"no runs for {a.problem} under {a.runs}")
    Gs = sorted({g for _, g in res})
    variants = [v for v in ORDER if any(k[0] == v for k in res)]
    G0 = a.G or max(Gs, key=lambda g: sum((v, g) in res for v in variants))

    print(f"\n{a.problem}, G = {G0}   (mean +- std over seeds; n = number of seeds)\n")
    print("| variant | n | seen | unseen | extrapolation | s / batch | rank-deficient states | chol fallbacks |")
    print("|---|---|---|---|---|---|---|---|")
    for v in variants:
        rs = res.get((v, G0), [])
        if not rs:
            continue
        cells = []
        for key in ("seen", "unseen", "extrapolation"):
            mu, sd, _ = stat(rs, key)
            cells.append(f"{mu:.4f}" + (f" +- {sd:.4f}" if np.isfinite(sd) else ""))
        rt, _, _ = stat(rs, "runtime_per_rollout_s")
        rd = sum(r["summary"].get("rank_deficient_states", 0) for r in rs)
        cf = sum(r["summary"].get("cholesky_fallbacks", 0) for r in rs)
        print(f"| {NAMES[v]} | {len(rs)} | {' | '.join(cells)} | {rt:.2f} | {rd} | {cf} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    T_train = None
    for v in variants:
        rs = res.get((v, G0), [])
        if not rs:
            continue
        t = np.array(rs[0]["t"])
        e = np.array([r["err_vs_t"] for r in rs])
        line, = ax[0].plot(t, e.mean(0), label=f"{NAMES[v]} (n={len(rs)})")
        if len(rs) > 1:
            ax[0].fill_between(t, e.min(0), e.max(0), color=line.get_color(), alpha=0.15, lw=0)
        ax[1].plot(t, np.mean([r["proj_err_vs_t"] for r in rs], 0), color=line.get_color(), label=NAMES[v])
        T_train = rs[0]["config"]["T_train"]
    for k in (0, 1):
        if T_train:
            ax[k].axvline(T_train, c="gray", ls=":")
        ax[k].set(yscale="log", xlabel="t")
        ax[k].grid(alpha=0.3, which="both")
        ax[k].legend(fontsize=7)
    ax[0].set(ylabel="relative L2 forecast error", title=f"{a.problem}: forecast error (G={G0}; band = seed range)")
    ax[1].set(ylabel="best relative L2 in current span", title="representation quality of the basis")
    for v in variants:
        for regime, ls in (("unseen", "-"), ("extrapolation", "--")):
            gh = [g for g in Gs if (v, g) in res]
            if gh:
                ax[2].plot(gh, [stat(res[(v, g)], regime)[0] for g in gh], ls, marker="o", label=f"{NAMES[v]}, {regime}")
    ax[2].set(xscale="log", yscale="log", xlabel="number of basis functions G", ylabel="relative L2",
              title="error vs number of basis functions")
    ax[2].set_xticks(Gs)
    ax[2].set_xticklabels(Gs)
    ax[2].grid(alpha=0.3, which="both")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    path = os.path.join(a.runs, f"compare_{a.problem}.png")
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()