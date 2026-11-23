"""Pick showcase caves for comparison videos from evaluation CSVs.

    python -m scripts.pick_showcase --csv runs/eval/v33c_memnav_cc_exit_memblind_episodes.csv \
        runs/eval/v33c_memnav_cc_none_memblind_episodes.csv --winner memnav_cc_exit --losers memnav_cc_none \
        --reference-csv runs/eval/memcaves_cue_episodes.csv --layouts rings hub

Finds test caves where the --winner policy reached the exit and every --losers policy did not (the clearest view of
what separates them). With --reference-csv, only caves the reference policy (default nav-oracle) solved count, so
the cave is known to be solvable. Prints `layout:seed` pairs on stdout, ready for `compare_videos --extra`, and a
short table on stderr. Policy names are the `policy` column of the CSVs: a run folder's name (memnav_cc_exit) or
nav-greedy / nav-oracle / nav-wall-left.
"""
import argparse
import csv
import sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="+", required=True, help="episode CSVs from scripts.eval_navigator")
    p.add_argument("--winner", required=True, help="policy that should succeed")
    p.add_argument("--losers", nargs="+", required=True, help="policies that should fail")
    p.add_argument("--layouts", nargs="*", default=None, help="default: every layout in the CSVs")
    p.add_argument("--reference-csv", nargs="*", default=[], help="CSVs with the reference policy's runs")
    p.add_argument("--reference", default="nav-oracle", help="policy that must have solved the cave")
    p.add_argument("--per-layout", type=int, default=1)
    args = p.parse_args()

    runs = defaultdict(dict)                                   # (layout, seed) -> policy -> (outcome, time)
    for f in args.csv + args.reference_csv:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            runs[(r["layout"], r["cave_seed"])][r["policy"]] = (r["outcome"], float(r["time"]))
    layouts = args.layouts or sorted({k[0] for k in runs})

    picks = []
    for lay in layouts:
        cands = []
        for (l, seed), pol in runs.items():
            if l != lay or pol.get(args.winner, ("",))[0] != "success":
                continue
            if args.reference_csv and pol.get(args.reference, ("",))[0] != "success":
                continue
            if all(pol.get(x, ("success",))[0] != "success" for x in args.losers) and all(x in pol for x in args.losers):
                cands.append((pol[args.winner][1], seed))
        cands.sort()                                           # fastest winner first: the shortest clear video
        if not cands:
            print(f"{lay}: no cave where {args.winner} succeeds and {', '.join(args.losers)} fail", file=sys.stderr)
        for t, seed in cands[: args.per_layout]:
            print(f"{lay}:{seed}  {args.winner} {t:.1f} s, {len(cands)} such caves", file=sys.stderr)
            picks.append(f"{lay}:{seed}")
    print(" ".join(picks))


if __name__ == "__main__":
    main()
