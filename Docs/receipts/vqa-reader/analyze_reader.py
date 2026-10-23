#!/usr/bin/env python3 -I
"""analyze_reader.py — U1.5 VQA transcription vs Apple Vision OCR as the text guard's reader.

Truth: SA-Text dataset transcriptions (satext/gt.json) for lv3 / lv1 / hq; the HR page's Vision OCR for ScreenSR
(the §11 pseudo-truth), plus U1's own HR read as a second ceiling. Per-word accounting follows the probe's
analyze_probe.py (tokens upper-cased, non-alphanumerics stripped, multiset match).

The guard question is precision: a reader that invents words it did not see is worse for a guard than one that
reads nothing. So beside word-F1 we report, per arm, INVENTED words (read but not in the truth) and the
operating point on the words Vision read confidently but wrongly: does U1 agree with Vision, with the truth, or
abstain?
"""
import json, os, re, statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = "/Volumes/Satechi/Development/mlxengine-forge/Tools/vosr-gates/20260926/ocr"
PROBE = f"{HERE}/../sr-probe"
SUBSET = {l.strip() for l in open(f"{PROBE}/satext/subset200.txt") if l.strip()}


def norm_tokens(strings):
    toks = []
    for s in strings:
        for w in re.split(r"\s+", s):
            w = re.sub(r"[^A-Z0-9]", "", w.upper())
            if w: toks.append(w)
    return toks


def ocr_strings(items):
    return [it["text"] if isinstance(it, dict) else it for it in items]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def per_image(ocr, keys=None, conf_min=None):
    out = {}
    for fname, items in ocr.items():
        k = fname[:-4]
        if keys is not None and k not in keys: continue
        if conf_min is not None:
            items = [it for it in items if isinstance(it, dict) and it.get("conf", 1) >= conf_min]
        out[k] = Counter(norm_tokens(ocr_strings(items)))
    return out


def score(truth, reads, keys):
    """word-level totals over the key set: matched / read / truth, image-mean F1, invented, abstained images."""
    matched = nread = ntruth = 0; f1s = []; invented = 0; empty = 0
    for k in keys:
        g = truth.get(k, Counter()); r = reads.get(k, Counter())
        m = sum((g & r).values()); matched += m; nread += sum(r.values()); ntruth += sum(g.values())
        invented += sum((r - g).values())
        if sum(r.values()) == 0: empty += 1
        if sum(g.values()):
            p = m / sum(r.values()) if sum(r.values()) else 0.0; rc = m / sum(g.values())
            f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    P = matched / nread if nread else float("nan"); R = matched / ntruth if ntruth else float("nan")
    return dict(f1=st.mean(f1s) if f1s else float("nan"), P=P, R=R, matched=matched, read=nread, truth=ntruth,
                invented=invented, empty=empty, n=len(keys))


def table(title, truth, arms, keys):
    print(f"\n── {title} (n = {len(keys)} images) ──")
    print(f"  {'reader':34} {'img-F1':>7} {'P':>7} {'R':>7} {'matched':>8} {'read':>6} {'truth':>6} {'INVENTED':>9} {'empty':>6}")
    for label, reads in arms:
        if reads is None: print(f"  {label:34} (missing)"); continue
        s = score(truth, reads, keys)
        print(f"  {label:34} {s['f1']:7.4f} {s['P']:7.4f} {s['R']:7.4f} {s['matched']:8} {s['read']:6} {s['truth']:6} {s['invented']:9} {s['empty']:6}")


def operating_point(title, truth, vision, u1, keys):
    """Words Vision read (confidently) that are NOT in the truth: what does U1 do on those images?"""
    agree = truth_hit = abstain = other = 0; imgs = 0
    for k in keys:
        g = truth.get(k, Counter()); v = vision.get(k, Counter()); u = u1.get(k, Counter())
        wrong = v - g
        if not wrong: continue
        imgs += 1
        for w, c in wrong.items():
            agree += min(u[w], c)              # U1 repeats Vision's wrong word
        truth_hit += sum((u & g).values())     # U1 reads truth words on that image
        if sum(u.values()) == 0: abstain += 1
    print(f"\n── {title}: images where Vision read ≥1 confident word not in the truth: {imgs} ──")
    print(f"  U1 repeats Vision's wrong words: {agree}   U1 reads truth words on those images: {truth_hit}   U1 abstains (reads nothing): {abstain} images")


gt = load(f"{PROBE}/satext/gt.json")
if gt:
    truth = {k: Counter(norm_tokens(v["text"])) for k, v in gt.items()}
    for lv, vis_name in (("lq3", "satext_lq3_bicubic"), ("lq1", "satext_lq1_bicubic"), ("hq", "satext_hq")):
        vis = load(f"{OLD}/{vis_name}.json"); u1 = load(f"{HERE}/reads/u1_satext_{lv}.json")
        if vis is None or u1 is None: continue
        keys = sorted(SUBSET & set(k[:-4] for k in u1))
        V1 = per_image(vis, keys, 1.0); U = per_image(u1, keys)
        # U1 with degenerate repetition collapsed: each distinct line once, each distinct word at most once per image
        Ud = {k: Counter(set(norm_tokens(sorted({it["text"] for it in u1.get(f"{k}.png", [])})))) for k in keys}
        # ensembles: words both readers agree on; Vision's certain words plus U1's words that Vision also read at any conf
        Vall = per_image(vis, keys)
        both = {k: (Vall.get(k, Counter()) & Ud.get(k, Counter())) for k in keys}
        confirm = {k: (V1.get(k, Counter()) | (Vall.get(k, Counter()) & Ud.get(k, Counter()))) for k in keys}
        arms = [("Vision (all lines)", Vall), ("Vision conf ≥ 1 (certain only)", V1),
                ("U1 VQA (raw)", U), ("U1 VQA, repeats collapsed", Ud),
                ("ENSEMBLE: Vision ∩ U1 (agreed words)", both), ("ENSEMBLE: Vision certain ∪ agreed", confirm)]
        table(f"SA-Text {lv} · words vs the dataset transcriptions", truth, arms, keys)
        operating_point(f"SA-Text {lv}", truth, per_image(vis, keys, 1.0), per_image(u1, keys), keys)
        # per-image agreement: both right / Vision only / U1 only / neither (image counted right when F1 == 1 on its words)
        V = per_image(vis, keys); U = per_image(u1, keys); both = vonly = uonly = neither = 0
        for k in keys:
            g = truth.get(k, Counter())
            if not sum(g.values()): continue
            vr = (g & V.get(k, Counter())) == g and (V.get(k, Counter()) - g) == Counter()
            ur = (g & U.get(k, Counter())) == g and (U.get(k, Counter()) - g) == Counter()
            both += vr and ur; vonly += vr and not ur; uonly += ur and not vr; neither += not vr and not ur
        print(f"  exact-image reads: both {both} · Vision only {vonly} · U1 only {uonly} · neither {neither}")

# ScreenSR: truth = HR page's Vision OCR (§11), and U1's HR read as the alternative ceiling
hr_vis = load(f"{PROBE}/ocr/screensr_hr.json") or load(f"{OLD}/screensr_hr.json")
lr_vis = load(f"{PROBE}/ocr/screensr_lr.json") or load(f"{OLD}/screensr_lr.json")
u1_lr = load(f"{HERE}/reads/u1_screensr_lr.json"); u1_hr = load(f"{HERE}/reads/u1_screensr_hr.json")
if hr_vis and u1_lr:
    pages = sorted(k[:-4] for k in u1_lr)
    T = per_image(hr_vis, pages); TU = per_image(u1_hr, pages) if u1_hr else {}
    print(f"\n── ScreenSR · per page: words read / matched against the HR page's VISION read (the §11 truth) ──")
    print(f"  {'page':6} {'truth':>6} | {'Vision LR read':>14} {'match':>6} {'F1':>6} | {'U1 LR read':>10} {'match':>6} {'F1':>6} {'INVENTED':>9} | {'U1 HR read':>10} {'match':>6}")
    for pg in pages:
        g = T.get(pg, Counter()); v = per_image(lr_vis, [pg]).get(pg, Counter()); u = per_image(u1_lr, [pg]).get(pg, Counter())
        uh = TU.get(pg, Counter())
        def f1(r):
            m = sum((g & r).values()); n = sum(r.values()); t = sum(g.values())
            if not t: return float("nan"), m
            p = m / n if n else 0.0; rc = m / t
            return (2 * p * rc / (p + rc) if p + rc else 0.0), m
        fv, mv = f1(v); fu, mu = f1(u); _, muh = f1(uh)
        print(f"  {pg:6} {sum(g.values()):6} | {sum(v.values()):14} {mv:6} {fv:6.3f} | {sum(u.values()):10} {mu:6} {fu:6.3f} {sum((u - g).values()):9} | {sum(uh.values()):10} {muh:6}")
    if u1_hr:
        print("\n  Chinese pages (001, 002): Vision HR read has no Latin tokens; U1's HR read, raw lines:")
        for pg in pages:
            if sum(T.get(pg, Counter()).values()) == 0:
                lines = ocr_strings(u1_hr.get(f"{pg}.png", []))
                print(f"    {pg}: {len(lines)} lines · " + " | ".join(lines)[:300])

# timing
print("\n── timing (MLX, 8step-v2 int8, M5 Max; Vision timings from VOSR-EVAL: first assess 25 s, then 0.05–0.18 s per page) ──")
for name in ("satext_lq3", "satext_lq1", "satext_hq", "screensr_lr", "screensr_hr"):
    p = f"{HERE}/logs/timing_{name}.json"
    if not os.path.exists(p): continue
    t = json.load(open(p))
    print(f"  {name:14} n={len(t):4}  median {st.median(x['seconds'] for x in t):6.2f} s/image  tokens median {st.median(x['tokens'] for x in t):5.0f}  max {max(x['tokens'] for x in t):5}")
