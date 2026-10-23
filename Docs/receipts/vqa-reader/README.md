# U1.5 VQA vs Apple Vision as the text guard's reader (2026-10-15)

_Receipt AB-R-0556 · Mac Studio M5 Max 128 GB · `sensenova-u1-swift` (release) · artifact `SenseNova-U1.5-8B-MoT-8step-v2-8bit`
(VQA runs on the understanding stream, bit-identical to the base tier — AB-L-0236) · no model or route changed._

**Question.** The SR text guard (MaskKit `TextLegibilityGuard`, ForgeCore `EngineTextGuardedUpscaler`, the N7
CORRUPTED/FIXED scorer) routes on Apple Vision OCR, which is English-only in practice and wrong on about a third of
its "certain" words on degraded input (AB-L-0173). The SR probe (AB-R-0555) closed U1.5 as an SR model but left
its reader untested. Does U1 VQA read degraded text more reliably than Vision — and, the guard's real question,
does it invent fewer words?

## What was added

`sensenova-cli --vqa-dir DIR --out reads.json [--vqa-prompt …] [--max-tokens N] [--list] [--limit] [--timing]`:
one model load per directory, every PNG through the und stream with one fixed prompt, answers written in the
`vosrgate ocr` JSON shape (`{file: [{"text": line, "conf": 1}]}`) so the probe's scorers apply unchanged. Prompt:

> Transcribe all the text in this image exactly as written, one line of the image per line of output. Do not
> describe the image. If there is no text, write NONE.

Greedy decoding, think block pre-closed (`Conversation.vqaPrompt(think: false)`), 256 tokens for crops, 1024 for
pages. A `NONE` answer is an empty read.

## Inputs and scoring

- **SA-Text lv3 / lv1 / hq**, the 200-image subset every learned arm ran on (128² signage crops; hq = the clean
  512² source). Truth = the dataset transcriptions. Vision rows = the 2026-09-26 OCR of the same files (the
  bicubic-×4 crops for lv3/lv1, as the guard sees them), with and without the "certain only" (conf = 1) filter.
- **ScreenSR** pages 001 / 002 (Chinese) / 010 (dense English) / 095 (sparse English), lr and hr. Truth = the HR
  page's Vision read (the §11 pseudo-truth); U1's own HR read is reported as the alternative ceiling.
- Per-word multiset accounting as in the probe (`analyze_probe.py`): matched / read / truth, image-mean F1,
  **INVENTED** = words read that are not in the truth, **empty** = images where the reader returned nothing, and
  the operating point on the words Vision read confidently but wrongly (does U1 repeat them, read the truth, or
  abstain). `run_reader.sh` is the chain; `analysis.txt` the tables; `reads/` the JSONs; `logs/` the timings.

## Results (`analysis.txt`; 200-image SA-Text subset per level, the four ScreenSR pages)

### Verdict: **no as a replacement, yes as a confirmer**

U1 VQA reads more of the truth than Vision at every degradation level, and invents far more. Its raw precision
(0.23 / 0.59 / 0.33 on lv3 / lv1 / hq) is below Vision's certain-only precision (0.82 / 0.73 / 0.78), so it cannot
be the guard's pseudo-truth on its own. But the two readers' errors are nearly disjoint: the words **both** read
are right 97–98 % of the time at 85–90 % of Vision's recall. That intersection is a protected-words set the guard
does not have today.

### SA-Text, words vs the dataset transcriptions (402 truth words per level)

| reader | lv3 P / R / invented | lv1 P / R / invented | hq P / R / invented |
|---|---|---|---|
| Vision, all lines | 0.685 / 0.092 / 17 | 0.694 / 0.209 / 37 | 0.735 / 0.587 / 85 |
| Vision, certain only (conf = 1) | 0.818 / 0.090 / 8 | 0.735 / 0.207 / 30 | 0.776 / 0.587 / 68 |
| U1 VQA, raw | 0.227 / 0.179 / 245 | 0.592 / 0.313 / 87 | 0.334 / 0.781 / 625 |
| U1 VQA, repeats collapsed | 0.462 / 0.179 / 84 | 0.601 / 0.311 / 83 | 0.685 / 0.741 / 137 |
| **Vision ∩ U1 (agreed words)** | **0.970 / 0.080 / 1** | **0.975 / 0.194 / 2** | **0.977 / 0.522 / 5** |

- **Recall:** U1 reads 72 / 126 / 314 truth words where Vision reads 37 / 84 / 236; it reads 20 / 26 / 45 images
  exactly right that Vision gets wrong, against 2 / 1 / 11 the other way. The VLM is the better reader of degraded
  signage, by a lot.
- **Invention, three kinds.** (1) Degenerate greedy loops: "5 | 5 | 5 …", "PARKING" × 30, "MIZUNO" × 25 — a
  decoding artifact, 60–80 % of the raw invented count on hq and lv3; collapsing repeated lines removes it
  (precision 0.33 → 0.69 on hq). (2) Describing instead of transcribing: "To transcribe the text in the image,
  I'll carefully examine…", or a paragraph about the Dom Tower after reading its sign. (3) Plausible wrong words on
  illegible crops: "CATHEDRAL OF THE SACRED HEART" for GOODERHAM WORTS, "C. E. M." for GUERLAIN — the AB-L-0172 class,
  now as text. (1) is fixable, (2) is mostly fixable with a stricter prompt, (3) is the reason it cannot be the
  truth.
- **Operating point on Vision's confident mistakes** (images where Vision read a certain word not in the truth:
  7 / 26 / 46): U1 repeats Vision's wrong word 1 / 2 / 5 times and reads truth words on those images 13 / 30 / 96
  times. It never abstains. The two readers fail on different images, which is what makes the intersection clean.

### ScreenSR pages (truth = the HR page's Vision read)

| page | truth words | Vision LR: read / matched / F1 | U1 LR: read / matched / F1 / invented |
|---|---:|---|---|
| 010 (dense) | 269 | 269 / 252 / 0.937 | 267 / 198 / 0.739 / 69 |
| 095 (sparse, low contrast) | 126 | 29 / 9 / 0.116 | 835 / 35 / 0.073 / 800 |
| 001, 002 | 0 | no tokens | no tokens ("NONE"; U1 describes 001 as a lion photograph) |

- On a dense legible page U1 reads fluent prose that is partly not on the page ("we'll go to see you for a while.
  'Last year, and henceforth…'"): 69 invented words where Vision has 17. On the sparse low-contrast page it
  confabulates 800 words of looping prose ("the parts are the parts of the parts…") where Vision reads 29. A VLM
  given a page it cannot read writes a page. For document pages Vision stays the reader.
- Pages 001 and 002 of the fixture set are photographs without text, not Chinese pages (the memo's note was
  wrong); the CJK question is untested here — no CJK page is in the fixtures.

### Cost

0.17–0.27 s per 128² crop and 4.4 s per 512² page (median; greedy, 1–5 tokens per crop, up to 1024 per page),
19.3 GB resident. Vision: 0.05–0.18 s per page after a 25 s first assess, no residency. U1 as a reader is only
free where U1 is already loaded.

### What this settles

1. **Do not replace Vision.** U1's precision is too low and its failure mode (fluent confabulation) is the worst
   possible for a guard.
2. **An agreement ensemble is worth building where U1 is resident:** protected words = Vision ∩ U1 after repeat
   collapse, precision 0.97–0.98 (Vision certain-only 0.73–0.82), recall 85–90 % of Vision's. It raises the N7
   scorer's and `wordProtection`'s word set from "a third wrong" to "one in forty wrong". Pages stay Vision-only.
3. **The reader is better than the renderer.** The SR probe (AB-R-0555) showed U1 cannot hold letters when
   generating; this shows it reads them better than Vision. If U1's text ever helps an upscale, it is as a reader
   feeding the guard, not as the SR model.
4. Decoding needs a repetition penalty or line dedupe before any such use; the stricter-prompt and CJK questions
   are open.
