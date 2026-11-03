"""
Compare runs of fa_evolve.py. Reads every runs/<problem>_<model>_G<G>[...]/results.json under --runs.

Figures (in --runs/compare_<problem>.png):
  left    forecast error vs time for each variant at one G (default: the largest G with all variants)
  middle  best-in-span projection error vs time (is the basis doing useful work?)
  right   error vs number of basis functions G, per variant and regime (the n-width figure)
Also prints a markdown table: seen / unseen / extrapolation error, runtime, rhs evaluations, parameters.

Usage:  python fa_compare.py --runs runs --problem kdv
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ORDER = ["fixed", "no_transport", "field_head", "full"]
NAMES = {"fixed": "fixed basis + FA", "no_transport": "moving + FA, no transport",
         "field_head": "moving + generic head", "full": "moving + FA + transport (full)"}


def load(runs, problem):
    out = {}
    for f in glob.glob(os.path.join(runs, f"{problem}_*", "results.json")):
        m = re.match(rf"{problem}_(fixed|no_transport|field_head|full)_G(\d+)", os.path.basename(os.path.dirname(f)))
        if m:
            out.setdefault((m.group(1), int(m.group(2))), []).append(json.load(open(f)))
    return out


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
    models = [m for m in ORDER if any(k[0] == m for k in res)]
    G0 = a.G or max([g for g in Gs if all((m, g) in res for m in models)] or Gs)

    print(f"\n{a.problem}, G = {G0}\n")
    print("| model | seen | unseen | extrapolation | s / rollout batch | rhs evals | params |")
    print("|---|---|---|---|---|---|---|")
    for m in models:
        rs = res.get((m, G0), [])
        if not rs:
            continue
        s = {k: np.mean([r["summary"][k] for r in rs]) for k in rs[0]["summary"]}
        print(f"| {NAMES[m]} | {s['seen']:.4f} | {s['unseen']:.4f} | {s['extrapolation']:.4f} | "
              f"{s['runtime_per_rollout_s']:.2f} | {s['rhs_evals_per_rollout']:.0f} | {rs[0]['n_params']:,} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
    for m in models:
        rs = res.get((m, G0), [])
        if not rs:
            continue
        t = np.array(rs[0]["t"])
        e = np.mean([r["err_vs_t"] for r in rs], 0)
        pe = np.mean([r["proj_err_vs_t"] for r in rs], 0)
        line, = ax[0].plot(t, e, label=NAMES[m])
        ax[1].plot(t, pe, color=line.get_color(), label=NAMES[m])
        T_train = rs[0]["config"]["T_train"]
    for k in (0, 1):
        ax[k].axvline(T_train, c="gray", ls=":")
        ax[k].set(yscale="log", xlabel="t")
        ax[k].grid(alpha=0.3, which="both")
        ax[k].legend(fontsize=7)
    ax[0].set(ylabel="relative L2 forecast error", title=f"{a.problem}: forecast error (G={G0})")
    ax[1].set(ylabel="best relative L2 in current span", title="representation quality of the basis")
    for m in models:
        for regime, ls in (("unseen", "-"), ("extrapolation", "--")):
            g_have = [g for g in Gs if (m, g) in res]
            if not g_have:
                continue
            v = [np.mean([r["summary"][regime] for r in res[(m, g)]]) for g in g_have]
            ax[2].plot(g_have, v, ls, marker="o", label=f"{NAMES[m]}, {regime}")
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