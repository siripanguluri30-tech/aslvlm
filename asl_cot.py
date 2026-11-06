# asl_cot.py -- prompt templates (direct vs chain-of-thought), longer-output clients, CoT-aware parsing.
import sys; sys.path.insert(0, "/content")
import os, re, json
import asl_quickstart as Q
import asl_ext as E

TEMPLATE_DIR = "/content/aslvlm/templates"
TEMPLATES = {
"direct.txt": "These frames are from a video of one American Sign Language sign.\nWhich candidate is being signed?\n\n{options}\n\nReply with only the label ({labels}).",
"cot_phonological.txt": (
"These frames, in time order, are from a video of one American Sign Language sign.\n\n"
"Step 1. Describe what you see: the handshape, where the hand is (location), and how it moves.\n"
"Step 2. Compare that description with each candidate and say which fit and which do not.\n\n"
"Candidates:\n{options}\n\n"
"Keep the reasoning under 120 words. Then end with one last line in exactly this form:\n"
"Final answer: <label>\nwhere <label> is one of {labels}."),
}

def write_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    for name, text in TEMPLATES.items():
        open(f"{TEMPLATE_DIR}/{name}", "w").write(text)
    return sorted(os.listdir(TEMPLATE_DIR))

# ---------- parsing (Task 2.3): the last "Final answer: X" line wins; reasoning text is never scanned ----------
_FINAL = re.compile(r'final\s*answer\s*[:\-]?\s*\**\(?([A-Za-z]|\d{1,2})\b', re.I)
_orig_parse = Q.parse

def smart_parse(raw, valid):
    if not isinstance(raw, str) or not raw.strip(): return None, "empty"
    hits = _FINAL.findall(raw)
    if hits:
        x = hits[-1].upper()
        return (x, "ok") if x in valid else (None, "unparseable")
    if len(raw.strip()) > 40:                      # long reasoning with no final line = cut off at the token limit
        return None, "truncated"
    return _orig_parse(raw, valid)

Q.parse = smart_parse                              # E.score_run looks Q.parse up at call time

# ---------- longer outputs without reloading the model ----------
class _Qwen:
    def __init__(self, c, n): self.c, self.n = c, n
    def generate(self, images, prompt):
        msgs = [{"role": "user", "content": [{"type": "image", "image": i} for i in images] + [{"type": "text", "text": prompt}]}]
        inp = self.c.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.c.model.device)
        out = self.c.model.generate(**inp, max_new_tokens=self.n, do_sample=False)
        return self.c.proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, out)], skip_special_tokens=True)[0]

class _Smol:
    def __init__(self, c, n): self.c, self.n = c, n
    def generate(self, images, prompt):
        import torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.c.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inp = self.c.proc(text=text, images=images, return_tensors="pt").to(self.c.model.device)
        with torch.no_grad():
            out = self.c.model.generate(**inp, max_new_tokens=self.n, do_sample=False)
        return self.c.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def long_client(client, max_new_tokens=256):
    k = type(client).__name__
    if k == "QwenLocalClient": return _Qwen(client, max_new_tokens)
    if k == "SmolVLM2LocalClient": return _Smol(client, max_new_tokens)
    raise ValueError("use a loaded local Qwen or SmolVLM2 client, got " + k)

# ---------- config-driven run with a prompt template (config field: prompt: templates/xxx.txt) ----------
def run_prompt(cfg, client, max_errors=3):
    import pandas as pd, hashlib
    cfg = E.resolve(cfg); cfg.setdefault("prompt", "templates/direct.txt")
    rid = E.run_id(cfg); pname = os.path.splitext(os.path.basename(cfg["prompt"]))[0]
    d = f"{E.RUNS}/{cfg['model']}_f{cfg['frames']['n']}_{cfg['frames']['strategy']}_k{cfg['task']['n_candidates']}_{pname}_{rid}"
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(f"{d}/config.resolved.json", "w"), indent=1)
    tpl = open(os.path.join(os.path.dirname(TEMPLATE_DIR), cfg["prompt"])).read()
    pred = f"{d}/predictions.jsonl"
    m = pd.read_csv(E.ROOT["manifest"]); m["norm_gt"] = m["Gloss"].str.lower().str.strip()
    ids = set(open(cfg["split"]).read().splitlines())
    df = m[m["Video file"].isin(ids)].sample(n=min(cfg["n_clips"], len(ids)), random_state=cfg["seed"]).reset_index(drop=True)
    vocab = sorted(m["norm_gt"].unique())
    done = {json.loads(l)["video"] for l in open(pred)} if os.path.exists(pred) else set()
    bad = 0
    with open(pred, "a") as fh:
        for _, r in df.iterrows():
            if r["Video file"] in done: continue
            item = Q.build_item(r["norm_gt"], vocab, cfg["task"]["n_candidates"], cfg["seed"], r["Video file"])
            labels = list(item["options"])
            prompt = tpl.format(options="\n".join(f"{l}. {o}" for l, o in item["options"].items()), labels=f"{labels[0]}-{labels[-1]}")
            fr = E.get_frames(os.path.join(E.ROOT["video_dir"], r["Video file"]), cfg["frames"]["strategy"], cfg["frames"]["n"], cfg["frames"]["max_side"])
            try: raw = client.generate(fr, prompt)
            except Exception as e: raw = f"__ERROR__ {e}"
            if raw.startswith("__ERROR__"):
                bad += 1; print("error:", raw[:150])
                if bad >= max_errors: print("Stopping; rerun later, finished clips are kept."); break
                continue
            bad = 0
            fh.write(json.dumps(dict(video=r["Video file"], signer=r["Participant ID"], gold=item["gold"], options=item["options"], raw=raw)) + "\n"); fh.flush()
    return pred

def status_rates(pred_path):
    rows = [json.loads(l) for l in open(pred_path)]
    c = {}
    for r in rows:
        s = Q.parse(r["raw"], set(r["options"]))[1]; c[s] = c.get(s, 0) + 1
    return {k: round(v / len(rows), 3) for k, v in c.items()}
