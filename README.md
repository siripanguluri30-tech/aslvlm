aslvlm: How well do general-purpose VLMs recognize American Sign Language?

A small, config-driven pipeline for testing vision-language models (VLMs) on isolated ASL signs from the ASL Citizen dataset.

Main finding so far: two 2B-parameter open models (SmolVLM2-2.2B-Instruct and Qwen3-VL-2B-Instruct) score at chance on this task, even when it is cut down to 4-way multiple choice. Frame count, frame sampling strategy and a chain-of-thought prompt made no measurable difference. A person got 18 of 20 on the same frames, so the frames carry enough information and the limit is the model. A 10-clip pilot of Gemini 3.8 Flash got 9 of 10, but the full run is not done yet.

Data
A stratified sample of 7,129 ASL Citizen clips (2,731 glosses, 30 signers), listed in data_sample_manifest.csv.
The videos are not in this repo. Get the dataset from the ASL Citizen authors and check its license before sharing any clips.
Evaluation uses a signer-independent test split: splits/test_signer_independent.txt (2,795 clips, 11 signers, none shared with train or validation).
ASL-LEX (for hard distractors)

Hard distractors need phonological coding. Download ASL-LEX 2.0 from https://osf.io/zpha4/ (Data Files, the sign table, saved as signdata.csv, read with encoding="latin-1") and put it at /content/signdata.csv. It is not redistributed in this repo; check its license (CC BY, cite both ASL-LEX papers: Caselli et al., 2017, Behavior Research Methods; Sevcikova Sehyr et al., 2021, Journal of Deaf Studies and Deaf Education). The manifest's ASL-LEX Code column links each clip to a row (7,113 of 7,129 clips have a code).

Task

For each test clip the model sees several frames and a multiple-choice question: the correct gloss plus k-1 distractors, labelled A-D (small k) or 1-50 (k above 26). Chance is 1/k. Distractors are random by default (distractors: random) or hard (distractors: hard). They are seeded per clip and per k, so every model gets identical questions. Numbered variants of the answer (for example axe1 for axe) are never used as distractors.

Files
File	                 What it does
asl_quickstart.py:	Baselines, split, frame extraction, prompt building, the answer parser, model clients (OpenAI-compatible, Gemini, local Qwen3-VL, local SmolVLM2, mock), inference and scoring
asl_ext.py:	Config-driven runs, three frame-sampling strategies, the frame cache, grid runs, scoring with bootstrap intervals and breakdowns
asl_cot.py:	Prompt templates (direct and chain-of-thought), a parser for "Final answer: X", longer-output wrapper for local models
asl_hard.py:	Hard distractors built from ASL-LEX 2.0 (same handshape and location, different movement)
direct.txt, cot_phonological.txt:	The two prompts (asl_cot.write_templates() writes them to templates/)
test_signer_independent.txt:	The versioned test split (the code reads it from splits/, see Setup)
data_sample_manifest.csv:	Metadata for the 7,129-clip sample

How a run works
1) A config (model, frames: n / strategy / max_side, task: n_candidates / distractors, split, n_clips, seed, prompt) is merged with defaults and saved as config.resolved.json next to the results.
2) Inference appends one JSON line per clip to predictions.jsonl (video, signer, gold label, options, raw reply) and skips clips already done, so a crashed run resumes. API errors are never saved as predictions.
3) Scoring is a separate step that reads those files. It reports accuracy, a clip-bootstrap 95% interval, a stricter signer-bootstrap interval, the unparseable rate, and breakdowns by signer, gloss frequency and clip duration.

Frame sampling strategies: uniform (whole clip), middle60 (drops the first and last 20%), motion (frames at equal slices of cumulative frame-to-frame change). Frames are cached under cache/frames/{video}/{strategy}_{n}_{max_side}/.

Answer parsing

A prediction is read as a label, never as free text. A bare label in any case is accepted, as are "Answer: C" and "C. word"; a lone lowercase "a" is treated as an article; two different valid labels count as ambiguous; no valid label counts as unparseable. For the chain-of-thought prompt only the last "Final answer: X" line is read, and a long reply with no such line is counted as truncated. Failures count as wrong and are reported next to accuracy.

Setup and running it (Google Colab, GPU runtime)
python
!git clone https://github.com/siripanguluri30-tech/aslvlm /content/aslvlm
!cp /content/aslvlm/asl_*.py /content/
import os, shutil
os.makedirs("/content/aslvlm/splits", exist_ok=True)
shutil.copy("/content/aslvlm/test_signer_independent.txt", "/content/aslvlm/splits/test_signer_independent.txt")
# the clips go in /content/aslvlm/data/ASL_Citizen_sample/videos (from the ASL Citizen dataset)

import sys; sys.path.insert(0, "/content")
import asl_quickstart as Q, asl_ext as E, asl_cot as C
C.write_templates()
client = Q.QwenLocalClient()                      # Qwen3-VL-2B-Instruct
paths = E.run_grid({"model": "qwen_local", "n_clips": 200}, client,
                   frames_n=(6,), strategies=("uniform",), ks=(4,))
print(E.table(paths))

# hard distractors (needs /content/signdata.csv)
hard = E.run_config({"model": "qwen_local", "n_clips": 200,
                     "task": {"n_candidates": 4, "distractors": "hard"}}, client)

For a Gemini run, store the key in the Colab secret GEMINI_API_KEY. Never paste a key into code or commit it.

Results so far (Qwen3-VL-2B unless noted; 200 test clips, k=4, chance 25%)
Condition	Accuracy
Random distractors, 6 uniform frames, direct prompt	29.0%
Hard distractors (ASL-LEX), same clips	29.0%
Chain-of-thought prompt (21.5% of replies truncated, counted wrong)	21.5%
1, 6 or 12 frames; uniform, middle 60% or motion-weighted sampling	28.5% to 30.0%
k=10 / k=50 (chance 10% / 2%)	14.0% / 2.5%
SmolVLM2-2.2B, k=4	28%
Human (blind, 20 clips, 6 frames), random / hard distractors	18/20 / 19/20

Every model-level result is within noise of chance, and blank-image controls scored the same as real frames, so the models do not appear to read the signs. The human scores show that the frames carry enough information.

Done: baselines, forced choice at k=4/10/50, signer-independent split, config-driven harness, frame and sampling sweep, controls, chain-of-thought prompt, hard distractors, per-signer / gloss-frequency / duration breakdowns, human blind tests.

Not done: the full 200-clip Gemini runs (needs a funded key; a 10-clip pilot got 9/10), SmolVLM2 at k=10 and k=50, a frame count of 3.

Limitations
200 clips per condition, so intervals are about ±6 points.
Three signers supply 70% of the test clips, and there are only 11 test signers.
The model run as "SmolVLM" is SmolVLM2-2.2B-Instruct, not SmolVLM-Instruct.
Isolated-sign recognition is not sign language translation.
The human blind test was done by a proficient hearing learner, not a Deaf signer.
