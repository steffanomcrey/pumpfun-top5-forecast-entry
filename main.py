#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: horizon-adaptive Monte Carlo forecaster.

Reads the house's 10-minute marks from $MAKEFUN_INPUT_DIR (else /input):
history.ndjson + task.json. Writes forecast.json to $MAKEFUN_OUTPUT_DIR
(else /output) in the make.fun forecast format: per-place coin probabilities,
a top5, and per-place market-cap size-bin probabilities.

Method (deterministic, standard library only, no network):
  * Per coin, from its mark-to-mark trajectory: last log10 market cap (SOL),
    a horizon-damped momentum drift, its own volatility, graduation state.
  * Per answer read (times taken from task.json, never hard-coded): simulate
    every candidate coin's future log cap (Gaussian core + a fat-tail jump
    mixture; curve coins also get a graduation-jump component), rank the
    simulation, and accumulate place frequencies and place-cap histograms.
  * A horizon-growing share of every place is left unnamed: it is the only
    mass that scores when a coin born after the cut (or never ranked in the
    history) takes the place, and the practice showed such coins taking
    2 of the top 5 at its last read.
"""
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
SIMS_PER_READ = 3000
MAX_CANDIDATES = 120
MIN_ROW_P = 0.0015         # drop probability rows below this; kept rows are rescaled to the place's named total
SIZE_NAMED_TOTAL = 0.98    # sizes leave 0.02 unnamed
TAU_DRIFT_H = 2.0          # momentum damping timescale, hours
GLOBAL_VOL_PER_SQRTH = 0.30  # fallback per-root-hour vol of log10 cap
JUMP_PROB = 0.12           # fat-tail mixture weight
JUMP_SCALE = 2.6           # fat-tail sd multiplier


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def logcap_sol(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    read_times = [parse_ts(s) for s in task["reads"]]

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark with rows")
    marks.sort(key=lambda m: parse_ts(m["at"]))
    t_first = parse_ts(marks[0]["at"])
    t_last = parse_ts(marks[-1]["at"])

    # ---- per-coin trajectories -------------------------------------------
    coins = {}
    for mi, m in enumerate(marks):
        t = parse_ts(m["at"])
        for r in m["rows"]:
            c = coins.setdefault(r["mint"], {"obs": [], "meta": r})
            c["obs"].append((t, logcap_sol(r), int(r["rank"])))
            c["meta"] = r
            c["last_mark"] = mi
    n_marks = len(marks)

    # market-wide drift of the leaders, log10 SOL per hour (damped later)
    lead_first = [logcap_sol(r) for r in marks[0]["rows"][:5]]
    lead_last = [logcap_sol(r) for r in marks[-1]["rows"][:5]]
    span_h = max((t_last - t_first).total_seconds() / 3600.0, 1e-6)
    market_drift = (sum(lead_last) / len(lead_last) - sum(lead_first) / len(lead_first)) / span_h
    market_drift = max(-0.05, min(0.10, market_drift))

    # global per-root-hour volatility from consecutive-mark changes
    diffs = []
    by_mark = []
    for m in marks:
        by_mark.append({r["mint"]: logcap_sol(r) for r in m["rows"]})
    for a, b in zip(by_mark, by_mark[1:]):
        dt_h = 1.0 / 6.0
        for mint in a.keys() & b.keys():
            diffs.append((b[mint] - a[mint]) / math.sqrt(dt_h))
    if len(diffs) >= 30:
        mean_d = sum(diffs) / len(diffs)
        global_vol = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / len(diffs))
        global_vol = max(0.15, min(0.60, global_vol))
    else:
        global_vol = GLOBAL_VOL_PER_SQRTH

    # ---- candidate features ----------------------------------------------
    cands = []
    for mint, c in coins.items():
        obs = c["obs"]
        t_end, last_log, last_rank = obs[-1]
        meta = c["meta"]
        # momentum: endpoint slope over the observed span, shrunk by span
        if len(obs) >= 2:
            t0, l0, _ = obs[0]
            span = max((t_end - t0).total_seconds() / 3600.0, 1e-6)
            slope = (last_log - l0) / span
            shrink = span / (span + 1.0)
            slope = max(-0.8, min(0.8, slope * shrink))
            steps = [obs[i + 1][1] - obs[i][1] for i in range(len(obs) - 1)]
            if len(steps) >= 3:
                ms = sum(steps) / len(steps)
                own_vol = math.sqrt(sum((s - ms) ** 2 for s in steps) / len(steps)) * math.sqrt(6.0)
                own_vol = max(0.12, min(0.70, own_vol))
            else:
                own_vol = global_vol
        else:
            slope, own_vol = 0.0, global_vol
        graduated = bool(meta.get("complete")) or meta.get("venue") in ("pool", "migrating")
        stale_marks = (n_marks - 1) - c["last_mark"]
        in_top30 = sum(1 for (_, _, rk) in obs if rk <= 30)
        persistence = in_top30 / max(1, len(obs))
        reserves_sol = int(meta.get("quote_reserves_lamports") or 0) / 1e9
        priority = last_log + 1.2 * max(0.0, slope) + (0.25 if graduated else 0.0) \
            + 0.15 * persistence - 0.35 * stale_marks
        cands.append({
            "mint": mint, "last_log": last_log, "slope": slope, "vol": own_vol,
            "graduated": graduated, "stale": stale_marks, "priority": priority,
            "reserves": reserves_sol, "best_rank": min(rk for (_, _, rk) in obs),
        })
    cands.sort(key=lambda c: (-c["priority"], c["mint"]))
    cands = cands[:MAX_CANDIDATES]

    horizons = [max((rt - t_last).total_seconds() / 3600.0, 0.05) for rt in read_times]
    h_mean = sum(horizons) / len(horizons)

    # unnamed (newcomer / outsider) share per place, growing with horizon
    unnamed = []
    for k in range(places):
        u = 0.16 + 0.038 * h_mean + 0.015 * k
        unnamed.append(max(0.20, min(0.60, u)))

    rng = random.Random(SEED)
    place_counts = [[0.0] * len(cands) for _ in range(places)]
    top5_counts = [0.0] * len(cands)
    exp_log = [0.0] * len(cands)
    size_counts = [[0.0] * (len(edges) + 1) for _ in range(places)]

    def bin_of(x):
        # bins are log10(cap SOL): bin 0 below edges[0], bin i in [edges[i-1], edges[i])
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    for H in horizons:
        damp = math.exp(-H / TAU_DRIFT_H)
        drift_t = H * damp                       # effective drift hours
        vol_t = math.sqrt(H)
        p_grad = 0.5 * (1.0 - math.exp(-H / 6.0))  # curve -> pool moon-jump chance
        mus, sds = [], []
        for c in cands:
            mu = c["last_log"] + (c["slope"] + market_drift) * drift_t
            mu -= 0.10 * c["stale"]              # stale sighting: fade it
            sd = c["vol"] * vol_t
            sd = sd * (0.85 if c["graduated"] else 1.10) + 0.05
            mus.append(mu)
            sds.append(sd)
        for _ in range(SIMS_PER_READ):
            sim = []
            for i, c in enumerate(cands):
                x = rng.gauss(mus[i], sds[i])
                u = rng.random()
                if u < JUMP_PROB:
                    x += rng.gauss(0.0, sds[i] * (JUMP_SCALE - 1.0))
                if not c["graduated"] and rng.random() < p_grad:
                    x += rng.gauss(1.0, 0.35)    # graduation repricing
                sim.append(x)
            order = sorted(range(len(cands)), key=lambda i: (-sim[i], cands[i]["mint"]))
            for k in range(places):
                w = order[k]
                place_counts[k][w] += 1.0
                size_counts[k][bin_of(sim[w])] += 1.0
            for w in order[:places]:
                top5_counts[w] += 1.0
            for i in range(len(cands)):
                exp_log[i] += sim[i]
    total = SIMS_PER_READ * len(horizons)

    probabilities = []
    for k in range(places):
        rows = []
        for i, c in enumerate(cands):
            p = place_counts[k][i] / total * (1.0 - unnamed[k])
            if p >= MIN_ROW_P:
                rows.append({"mint": c["mint"], "place": k + 1, "p": p})
        rows.sort(key=lambda r: (-r["p"], r["mint"]))
        rows = rows[:100]
        kept = sum(r["p"] for r in rows)
        if kept > 0:
            # rescale so the named mass is exactly the place's intended total:
            # the threshold's dropped mass is redistributed, not lost to unnamed
            scale = (1.0 - unnamed[k]) / kept
            for r in rows:
                r["p"] = round(r["p"] * scale, 6)
        probabilities.extend(rows)

    sizes = []
    for k in range(places):
        counts = size_counts[k]
        tot = sum(counts) or 1.0
        probs = [n / tot for n in counts]
        # light neighbour smoothing for a fatter, honest size distribution
        sm = []
        for i in range(len(probs)):
            v = 0.5 * probs[i]
            v += 0.25 * (probs[i - 1] if i > 0 else 0.0)
            v += 0.25 * (probs[i + 1] if i < len(probs) - 1 else 0.0)
            sm.append(v)
        rows = []
        for b, p in enumerate(sm):
            p = p * SIZE_NAMED_TOTAL
            if p >= 0.0015:
                rows.append({"place": k + 1, "bin": b, "p": p})
        kept = sum(r["p"] for r in rows)
        if kept > 0:
            scale = SIZE_NAMED_TOTAL / kept
            for r in rows:
                r["p"] = round(r["p"] * scale, 6)
        sizes.extend(rows)

    order_idx = sorted(range(len(cands)),
                       key=lambda i: (-top5_counts[i], -exp_log[i], cands[i]["mint"]))
    top5 = [cands[i]["mint"] for i in order_idx[:places]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
