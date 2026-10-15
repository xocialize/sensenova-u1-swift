#!/usr/bin/env python3 -I
"""analyze_probe.py — the VOSR-EVAL §7 gate tables for the SenseNova-U1.5 zero-training SR probe
(SENSENOVA-SR-EVAL §7 step 1), beside the 2026-09-26 arms (VOSR2, SeedVR2-3B, RealPLKSR, Real-ESRGAN) and the
2026-10-09 Iris-3B rows. Adapted from WIP/iris-3b-oracle/gates/analyze_iris.py; the classical-arm rows are
checked to reproduce the September run so the merge is valid.

Probe arms are every `__U1-*.png` method suffix found in bench/dump. Sections are skipped while their inputs are
missing. Run from this directory:  python3 -I analyze_probe.py [--arms U1-a-t0.7-s1,...]
"""
import csv, glob, json, os, re, statistics as st, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = "/Volumes/Satechi/Development/mlxengine-forge/Tools/vosr-gates/20260926"
IRIS = "/Volumes/Satechi/Development/mlxengine-image/WIP/iris-3b-oracle/gates"
OLD_METHODS = ["VOSR2", "SeedVR2-3B", "RealPLKSR", "Real-ESRGAN"]
IRIS_METHODS = ["Iris-3B"]
CLASSICAL = ["bicubic-cg", "lanczos-mps"]
regimes, classes = ["bicubic", "lanczos", "jpeg"], ["photo", "graphic"]


def mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def load_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k not in ("image", "class", "regime", "method"):
                try: r[k] = float(r[k])
                except ValueError: r[k] = float("nan")
    return rows


arms_arg = None
if "--arms" in sys.argv:
    arms_arg = sys.argv[sys.argv.index("--arms") + 1].split(",")
dump = f"{HERE}/bench/dump"
probe_methods = sorted({re.sub(r".*__", "", f)[:-4] for f in os.listdir(dump) if "__U1-" in f}) if os.path.exists(dump) else []
if arms_arg: probe_methods = [m for m in probe_methods if m in arms_arg]
print(f"probe arms: {probe_methods}")

# ---------------------------------------------------------------- G1 / G3 / detail
new_csv = f"{HERE}/bench/score.csv"
rows, methods = [], []
if os.path.exists(new_csv):
    new = load_rows(new_csv)
    old = load_rows(f"{OLD}/score.csv")
    iris = load_rows(f"{IRIS}/bench/score.csv") if os.path.exists(f"{IRIS}/bench/score.csv") else []
    mism = 0
    for r in new:
        if r["method"] in CLASSICAL + ["REFERENCE"]:
            o = [x for x in old if (x["image"], x["regime"], x["method"]) == (r["image"], r["regime"], r["method"])]
            if o and (abs(o[0]["fr"] - r["fr"]) > 0.05 or abs(o[0]["grad"] - r["grad"]) > 0.005):
                mism += 1; print(f"  INPUT MISMATCH {r['image']}/{r['regime']}/{r['method']}: FR {r['fr']:.2f} vs {o[0]['fr']:.2f}")
    print(f"input identity vs 2026-09-26: classical-arm/reference mismatches = {mism} (0 = same crops, same regimes)")
    rows = ([r for r in new if r["method"] in probe_methods + CLASSICAL + ["REFERENCE"]]
            + [r for r in old if r["method"] in OLD_METHODS] + [r for r in iris if r["method"] in IRIS_METHODS])
    methods = [m for m in probe_methods + OLD_METHODS + IRIS_METHODS + CLASSICAL if any(r["method"] == m for r in rows)]

    def table(title, key, fmt="{:8.2f}", ms=None, ref=False):
        ms = ms or methods
        if ref: ms = ms + ["REFERENCE"]
        print(f"\n── {title} ──")
        print("  regime   class    " + "".join(f"{m:>16}" for m in ms))
        for rg in regimes:
            for c in classes:
                vals = [mean([r[key] for r in rows if r["regime"] == rg and r["class"] == c and r["method"] == m]) for m in ms]
                print(f"  {rg:8} {c:8} " + "".join(f"{fmt.format(v):>16}" for v in vals))

    table("G1 · FULL-REFERENCE SSIMULACRA2 vs the lossless 2048² reference (higher better)", "fr")
    table("DETAIL · mean gradient magnitude (HF content, NOT quality)", "grad", "{:8.3f}", ref=True)
    table("G3 · FidelityGuard downsample-consistency SSIM, WORST 64px tile (higher = less invented)", "dc_worst", "{:8.4f}", ref=True)
    table("G3 · downsample-consistency SSIM, mean", "dc_ssim", "{:8.4f}", ref=True)
    table("OCTAVE 0 SSIM vs reference (finest band)", "oct0", "{:8.4f}")
    table("BASE-BAND SSIM vs reference (coarsest — colour drift shows here)", "oct_base", "{:8.4f}")
    print("\n── per-image FR, jpeg regime (the regime that decides G1) ──")
    imgs = sorted({r["image"] for r in rows})
    print("  image                  " + "".join(f"{m:>16}" for m in methods))
    for img in imgs:
        vals = [mean([r["fr"] for r in rows if r["image"] == img and r["regime"] == "jpeg" and r["method"] == m]) for m in methods]
        print(f"  {img:22} " + "".join(f"{v:16.2f}" for v in vals))

    def g1(m, c):
        return mean([r["fr"] for r in rows if r["regime"] == "jpeg" and r["class"] == c and r["method"] == m])
    print()
    for m in probe_methods:
        verdict = "PASS" if g1(m, "photo") > max(g1("lanczos-mps", "photo"), g1("RealPLKSR", "photo")) else "FAIL"
        print("  G1 jpeg/photo:   %-18s %6.2f vs lanczos %.2f / RealPLKSR %.2f / VOSR2 %.2f  →  %s" % (
            m, g1(m, "photo"), g1("lanczos-mps", "photo"), g1("RealPLKSR", "photo"), g1("VOSR2", "photo"), verdict))
        print("  G1 jpeg/graphic: %-18s %6.2f vs lanczos %.2f / RealPLKSR %.2f / VOSR2 %.2f" % (
            m, g1(m, "graphic"), g1("lanczos-mps", "graphic"), g1("RealPLKSR", "graphic"), g1("VOSR2", "graphic")))

# ---------------------------------------------------------------- NR
nr_new = f"{HERE}/bench/nr.json"
if os.path.exists(nr_new) and rows:
    nr = {}
    for p in (f"{OLD}/nr.json", f"{IRIS}/bench/nr.json", nr_new):
        if os.path.exists(p):
            nr.update({k.split("/", 1)[1] if "/" in k else k: v for k, v in json.load(open(p)).items()})
    print("\n── NO-REFERENCE · SigLIP2 NR-IQA (0–1, high = clean; low-input is 512², all else 2048²) ──")
    print("  regime   class    " + "".join(f"{m:>16}" for m in methods) + f"{'REF(2048²)':>12}{'LOW(512²)':>12}")
    for rg in regimes:
        for c in classes:
            imgs_c = sorted({r["image"] for r in rows if r["class"] == c})
            vals = [mean([nr.get(f"{i}__{rg}__{m}.png", float("nan")) for i in imgs_c]) for m in methods]
            ref = mean([nr.get(f"{i}__reference.png", float("nan")) for i in imgs_c])
            low = mean([nr.get(f"{i}__{rg}__low.png", float("nan")) for i in imgs_c])
            print(f"  {rg:8} {c:8} " + "".join(f"{v:16.4f}" for v in vals) + f"{ref:12.4f}{low:12.4f}")


# ---------------------------------------------------------------- grid measure (32 px = U1 token, 16 px = Iris patch)
def grid_ratio(path, period):
    """Energy of the luma gradient spectrum at the period's harmonics vs the neighbouring bins (1 = no grid)."""
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    gx = np.diff(im, axis=1)[:-1, :]; gy = np.diff(im, axis=0)[:, :-1]
    g = np.hypot(gx, gy)
    g = g - g.mean()
    F = np.abs(np.fft.rfft2(g)) ** 2
    H, W = g.shape
    ratios = []
    for axis in (0, 1):
        n = H if axis == 0 else W
        prof = F.sum(axis=1 - axis) if axis == 0 else F.sum(axis=0)
        peak, side = 0.0, 0.0
        for k in range(1, period // 2 + 1):
            f = k * n // period
            if axis == 0:
                idx = [f % n, (-f) % n]
            else:
                if f >= prof.shape[0]: continue
                idx = [f]
            for i in idx:
                peak += prof[i]
                nb = [j for j in (i - 3, i - 2, i + 2, i + 3) if 0 <= j < prof.shape[0]]
                side += st.mean([prof[j] for j in nb]) if nb else 0
        ratios.append(peak / side if side else float("nan"))
    return st.mean(ratios)


if os.path.exists(dump) and probe_methods:
    try:
        import numpy, PIL  # noqa
        for period in (32, 16):
            ms = probe_methods + ["bicubic-cg", "lanczos-mps"]
            print(f"\n── PATCH GRID · gradient-spectrum energy at the {period}-px harmonics vs neighbours (1.0 = none; jpeg regime; mean over 27 → 9 images) ──")
            print("  image                  " + "".join(f"{m:>16}" for m in ms) + f"{'REFERENCE':>12}{'VOSR2':>10}{'Iris-3B':>10}")
            tot = {m: [] for m in ms + ["REFERENCE", "VOSR2", "Iris-3B"]}
            for img in sorted({f.split("__")[0] for f in os.listdir(dump) if f.endswith("__reference.png")}):
                vals = []
                for m in ms:
                    p = f"{dump}/{img}__jpeg__{m}.png"
                    v = grid_ratio(p, period) if os.path.exists(p) else float("nan"); vals.append(v); tot[m].append(v)
                extras = []
                for m, p in (("REFERENCE", f"{dump}/{img}__reference.png"), ("VOSR2", f"{OLD}/dump/{img}__jpeg__VOSR2.png"),
                             ("Iris-3B", f"{IRIS}/bench/dump/{img}__jpeg__Iris-3B.png")):
                    v = grid_ratio(p, period) if os.path.exists(p) else float("nan"); extras.append(v); tot[m].append(v)
                print(f"  {img:22} " + "".join(f"{v:16.2f}" for v in vals) + "".join(f"{v:10.2f}" for v in extras[:1]) + "".join(f"{v:10.2f}" for v in extras[1:]))
            print(f"  {'MEAN':22} " + "".join(f"{mean(tot[m]):16.2f}" for m in ms) + f"{mean(tot['REFERENCE']):12.2f}{mean(tot['VOSR2']):10.2f}{mean(tot['Iris-3B']):10.2f}")
    except ImportError:
        print("\n(grid measure skipped: numpy/PIL not importable)")


# ---------------------------------------------------------------- G2 text (SA-Text)
def norm_tokens(strings):
    toks = []
    for s in strings:
        for w in re.split(r"\s+", s):
            w = re.sub(r"[^A-Z0-9]", "", w.upper())
            if w: toks.append(w)
    return toks


def f1_sets(gt_tokens, rec_tokens):
    g, r = Counter(gt_tokens), Counter(rec_tokens)
    matched = sum((g & r).values())
    if not gt_tokens: return None
    p = matched / len(rec_tokens) if rec_tokens else 0.0
    rc = matched / len(gt_tokens)
    return (2 * p * rc / (p + rc)) if (p + rc) else 0.0, p, rc


def ocr_strings(items):
    return [it["text"] if isinstance(it, dict) else it for it in items]


def evaluate(ocr_json, gt_map, subset=None):
    ocr = json.load(open(ocr_json))
    f1s, ps, rs = [], [], []
    for fname, items in ocr.items():
        key = fname[:-4]
        if key not in gt_map or (subset is not None and key not in subset): continue
        res = f1_sets(norm_tokens(gt_map[key]), norm_tokens(ocr_strings(items)))
        if res is None: continue
        f1s.append(res[0]); ps.append(res[1]); rs.append(res[2])
    return (st.mean(f1s), st.mean(ps), st.mean(rs), len(f1s)) if f1s else (float("nan"),) * 3 + (0,)


def per_image_sets(p):
    o = json.load(open(p))
    return {k[:-4]: Counter(norm_tokens(ocr_strings(v))) for k, v in o.items()}


def corrupted_fixed(gt, inp_sets, out_sets, keys=None):
    corrupted = fixed = in_ok = out_ok = n = 0
    for k, words in gt.items():
        if k not in out_sets or k not in inp_sets or (keys is not None and k not in keys): continue
        n += 1
        g = Counter(norm_tokens(words))
        for w, c in g.items():
            i_ok = min(inp_sets[k][w], c); o_ok = min(out_sets[k][w], c)
            in_ok += i_ok; out_ok += o_ok
            corrupted += max(i_ok - o_ok, 0); fixed += max(o_ok - i_ok, 0)
    return in_ok, out_ok, corrupted, fixed, n


gt_path = f"{HERE}/satext/gt.json"
probe_ocr = sorted(glob.glob(f"{HERE}/ocr/satext_*_lq*.json"))
if os.path.exists(gt_path) and probe_ocr:
    gt = {k: v["text"] for k, v in json.load(open(gt_path)).items()}
    print("\n── G2 · SA-Text word-F1 of Apple Vision OCR vs the dataset transcriptions (lv3 = heaviest; 128² → 512²) ──")
    subset = {k[:-4] for k in json.load(open(f"{OLD}/ocr/satext_seedvr2_lq3.json"))}
    for lv in ("lv3", "lv1"):
        lq = lv.replace("lv", "lq")
        arms = [("HQ ceiling (2026-09-26 OCR)", f"{OLD}/ocr/satext_hq.json"),
                (f"LQ {lv} bicubic ×4 (un-restored)", f"{OLD}/ocr/satext_{lq}_bicubic.json"),
                (f"VOSR2 {lv}", f"{OLD}/ocr/satext_vosr2_{lq}.json"),
                (f"Iris-3B {lv}", f"{IRIS}/ocr/satext_iris_{lq}.json")]
        if lv == "lv3":
            arms += [("SeedVR2-3B lv3 (200)", f"{OLD}/ocr/satext_seedvr2_lq3.json"), ("Real-ESRGAN lv3 (200)", f"{OLD}/ocr/satext_realesrgan_lq3.json"),
                     ("RealPLKSR lv3 (200)", f"{OLD}/ocr/satext_realplksr_lq3.json")]
        ours = [(f"{os.path.basename(p)[7:-(len(lq)+6)]} {lv}", p) for p in probe_ocr
                if p.endswith(f"_{lq}.json") and json.load(open(p))]   # an empty json = a stage that did not run
        arms += ours
        # every probe arm's image set (the sweep may have run on the 200-subset only)
        probe_keys = None
        for _, p in ours:
            ks = {k[:-4] for k in json.load(open(p))}
            probe_keys = ks if probe_keys is None else probe_keys & ks
        for title, sub in (("all images each arm ran on", None), (f"the {len(subset)}-image subset every learned arm ran on", subset),
                           (f"the {len(probe_keys or [])}-image set every PROBE arm ran on", probe_keys)):
            if sub is None and title.startswith("the"): continue
            print(f"  ── {lv} · {title} ──")
            print(f"  {'arm':50} {'F1':>7} {'P':>7} {'R':>7} {'n':>5}")
            for label, p in arms:
                if not os.path.exists(p): continue
                f1, pr, rc, n = evaluate(p, gt, sub)
                print(f"  {label:50} {f1:7.4f} {pr:7.4f} {rc:7.4f} {n:5}")
        inp = f"{OLD}/ocr/satext_{lq}_bicubic.json"
        if os.path.exists(inp):
            I = per_image_sets(inp)
            print(f"  ── {lv} · GT words: input-correct → CORRUPTED by the arm / input-wrong → FIXED by the arm (on the probe-arm image set, n = {len(probe_keys or [])}) ──")
            for label, p in [("VOSR2", f"{OLD}/ocr/satext_vosr2_{lq}.json"), ("Iris-3B", f"{IRIS}/ocr/satext_iris_{lq}.json")] + ours:
                if not os.path.exists(p): continue
                in_ok, out_ok, corrupted, fixed, n = corrupted_fixed(gt, I, per_image_sets(p), probe_keys)
                print(f"    {label:28} n {n:4}  input-correct {in_ok:5}  output-correct {out_ok:5}  CORRUPTED {corrupted:4}  FIXED {fixed:4}")

# ---------------------------------------------------------------- ScreenSR (word-F1 vs the HR page's OCR, the §11 scorer)
scr = sorted(glob.glob(f"{HERE}/ocr/screensr_*.json"))
hr_ocr = f"{HERE}/ocr/screensr_hr.json" if os.path.exists(f"{HERE}/ocr/screensr_hr.json") else f"{OLD}/ocr/screensr_hr.json"
if scr and os.path.exists(hr_ocr):
    pages = sorted(os.path.splitext(f)[0] for f in os.listdir(f"{HERE}/screensr/lr") if f.endswith(".png"))
    H = {k[:-4]: norm_tokens(ocr_strings(v)) for k, v in json.load(open(hr_ocr)).items()}
    print(f"\n── ScreenSR · Vision word-F1 vs the HR page's OCR (pages {', '.join(pages)}; 512² pages; the §11 scorer) ──")
    arms = [("input ×1 (2026-09-26 OCR)", f"{OLD}/ocr/screensr_lr.json"), ("VOSR2 ×1 (2026-09-26)", f"{OLD}/ocr/screensr_vosr2.json")]
    arms += [(os.path.basename(p)[9:-5], p) for p in scr if "_hr" not in p and "_lr" not in p]
    if os.path.exists(f"{HERE}/ocr/screensr_lr.json"): arms.insert(1, ("input ×1 (this run's OCR)", f"{HERE}/ocr/screensr_lr.json"))
    print(f"  {'arm':34}" + "".join(f"{p:>8}" for p in pages) + f"{'mean':>8}")
    inp_sets = {k[:-4]: Counter(norm_tokens(ocr_strings(v))) for k, v in json.load(open(f"{OLD}/ocr/screensr_lr.json")).items()}
    for label, p in arms:
        o = json.load(open(p))
        vals = []
        for pg in pages:
            items = o.get(f"{pg}.png")
            if items is None or pg not in H: vals.append(float("nan")); continue
            r = f1_sets(H[pg], norm_tokens(ocr_strings(items)))
            vals.append(r[0] if r else float("nan"))
        # CORRUPTED / FIXED vs the un-restored input, HR OCR as truth
        O = {k[:-4]: Counter(norm_tokens(ocr_strings(v))) for k, v in o.items()}
        corrupted = fixed = 0
        for pg in pages:
            if pg not in O or pg not in inp_sets: continue
            for w, c in Counter(H[pg]).items():
                i_ok = min(inp_sets[pg][w], c); o_ok = min(O[pg][w], c)
                corrupted += max(i_ok - o_ok, 0); fixed += max(o_ok - i_ok, 0)
        print(f"  {label:34}" + "".join(f"{v:8.3f}" for v in vals) + f"{mean(vals):8.3f}   CORRUPTED {corrupted:3}  FIXED {fixed:3}")

# ---------------------------------------------------------------- G5 timing
tim = sorted(glob.glob(f"{HERE}/logs/timing_*.json"))
if tim:
    print("\n── G5 · timing (this probe: MLX, 8step-v2 int8, M5 Max; prior arms PyTorch MPS) ──")
    for p in tim + [f"{OLD}/timing_corpus.json", f"{OLD}/timing_satext3.json", f"{IRIS}/logs/timing_corpus.json", f"{IRIS}/logs/timing_satext3.json"]:
        if not os.path.exists(p): continue
        t = json.load(open(p))
        if not t: continue
        label = os.path.relpath(p, HERE) if p.startswith(HERE) else p.replace("/Volumes/Satechi/Development/", "")
        print(f"  {label:70} n={len(t):4}  median {st.median(x['seconds'] for x in t):6.2f} s/image · forward {st.median(x['dit_ms_per_call'] for x in t):6.0f} ms × {st.median(x['dit_forwards'] for x in t):.0f}")
