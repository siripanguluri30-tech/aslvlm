CONFIG = dict(
    root        = "/content/aslvlm",
    video_dir   = "/content/aslvlm/data/ASL_Citizen_sample/videos",
    manifest    = "/content/aslvlm/data/data_sample_manifest.csv",
    split_file  = "/content/aslvlm/splits/test_signer_independent.txt",
    n_clips     = 200,
    k           = 4,
    n_frames    = 6,
    max_side    = 384,
    seed        = 0,
    out_dir     = "/content/outputs",
    model       = "mock",   # "mock" | "smolvlm2_local" | "smolvlm" | "qwen_local" | "gemini"
    smolvlm_endpoint = "http://YOUR_ENDPOINT/v1",
    smolvlm_name     = "HuggingFaceTB/SmolVLM-Instruct",
    gemini_model     = "CHECK AI STUDIO FOR CURRENT MODEL NAME",
)

import os, re, io, json, base64, random
import numpy as np, pandas as pd, cv2
from PIL import Image

def make_split_and_baselines(cfg):
    m = pd.read_csv(cfg["manifest"])
    S = {sp: set(g["Participant ID"]) for sp, g in m.groupby("split")}
    print("Signer overlap test/train:", len(S["test"] & S["train"]), "| test/val:", len(S["test"] & S["val"]))
    test = m[m["split"] == "test"]
    os.makedirs(os.path.dirname(cfg["split_file"]), exist_ok=True)
    test["Video file"].to_csv(cfg["split_file"], index=False, header=False)
    g = test["Gloss"].str.lower().value_counts()
    print(f"test clips: {len(test)} | test signers: {test['Participant ID'].nunique()} | glosses: {len(g)}")
    print(f"random baseline (full vocab): {1/m['Gloss'].nunique():.4%}")
    print(f"majority-gloss baseline: {g.iloc[0]/len(test):.4%}  (always '{g.index[0]}')")
    for k in (4, 10, 50): print(f"random baseline at k={k}: {1/k:.1%}")
    print(test["Participant ID"].value_counts().to_string())

def load_clips(cfg):
    m = pd.read_csv(cfg["manifest"])
    m["norm_gt"] = m["Gloss"].str.lower().str.strip()
    ids = set(open(cfg["split_file"]).read().splitlines())
    df = m[m["Video file"].isin(ids)][["Video file", "Participant ID", "norm_gt"]]
    df = df.sample(n=min(cfg["n_clips"], len(df)), random_state=cfg["seed"]).reset_index(drop=True)
    return df, sorted(m["norm_gt"].unique())

def extract_frames(path, n, max_side):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok: continue
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        im.thumbnail((max_side, max_side))
        frames.append(im)
    cap.release()
    return frames

def make_labels(k):
    return [chr(65+i) for i in range(k)] if k <= 26 else [str(i+1) for i in range(k)]

def build_item(correct, vocab, k, seed, video_id):
    rng = random.Random(f"{seed}-{video_id}-{k}")
    base = re.sub(r"\d+$", "", correct)
    pool = [v for v in vocab if re.sub(r"\d+$", "", v) != base]
    opts = rng.sample(pool, k-1) + [correct]
    rng.shuffle(opts)
    labels = make_labels(k)
    body = "\n".join(f"{l}. {o}" for l, o in zip(labels, opts))
    prompt = (f"These frames are from a video of one American Sign Language sign.\n"
              f"Which candidate is being signed?\n\n{body}\n\n"
              f"Reply with only the label ({labels[0]}-{labels[-1]}).")
    return dict(prompt=prompt, gold=labels[opts.index(correct)], options=dict(zip(labels, opts)))

def parse(raw, valid):
    if not isinstance(raw, str) or not raw.strip(): return None, "empty"
    t = raw.strip()
    m = re.match(r'^\W*(?:option|answer|choice)?\W*([A-Za-z]|\d{1,2})\W*$', t, re.I) \
        or re.search(r'(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z]|\d{1,2})\b', t, re.I)
    if m:
        x = m.group(1).upper()
        return (x, "ok") if x in valid else (None, "unparseable")
    found = {x for x in re.findall(r'\b([A-Z]|\d{1,2})\b', t) if x in valid}
    if len(found) == 1: return found.pop(), "ok"
    return (None, "ambiguous") if found else (None, "unparseable")

def _b64(im):
    b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode()

class OpenAICompatClient:
    def __init__(self, base_url, model):
        from openai import OpenAI
        self.c, self.model = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "none")), model
    def generate(self, images, prompt):
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(i)}"}} for i in images]
        content.append({"type": "text", "text": prompt})
        r = self.c.chat.completions.create(model=self.model, temperature=0, max_tokens=20,
                                           messages=[{"role": "user", "content": content}])
        return r.choices[0].message.content

class GeminiClient:
    def __init__(self, model):
        from google import genai
        self.genai, self.model = genai, model
        self.c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    def generate(self, images, prompt):
        from google.genai import types
        parts = [types.Part.from_bytes(data=base64.b64decode(_b64(i)), mime_type="image/jpeg") for i in images]
        r = self.c.models.generate_content(model=self.model, contents=parts + [prompt],
                                           config=types.GenerateContentConfig(temperature=0, max_output_tokens=20))
        return r.text or ""

class QwenLocalClient:
    def __init__(self, name="Qwen/Qwen3-VL-2B-Instruct"):
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=torch.float16, device_map="auto")
    def generate(self, images, prompt):
        msgs = [{"role": "user", "content": [{"type": "image", "image": i} for i in images] + [{"type": "text", "text": prompt}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=20, do_sample=False)
        return self.proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, out)], skip_special_tokens=True)[0]

class MockClient:
    def generate(self, images, prompt):
        return random.choice("ABCD")

class SmolVLM2LocalClient:
    def __init__(self, name="HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    def generate(self, images, prompt):
        import torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inp = self.proc(text=text, images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=20, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def get_client(cfg):
    m = cfg["model"]
    if m == "mock":           return MockClient()
    if m == "smolvlm2_local": return SmolVLM2LocalClient()
    if m == "smolvlm":        return OpenAICompatClient(cfg["smolvlm_endpoint"], cfg["smolvlm_name"])
    if m == "gemini":         return GeminiClient(cfg["gemini_model"])
    if m == "qwen_local":     return QwenLocalClient()
    raise ValueError(m)

def run_inference(cfg, client):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    df, vocab = load_clips(cfg)
    out = f'{cfg["out_dir"]}/pred_{cfg["model"]}_k{cfg["k"]}_f{cfg["n_frames"]}.jsonl'
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["video"] for l in open(out)}
    with open(out, "a") as fh:
        for _, r in df.iterrows():
            if r["Video file"] in done: continue
            item = build_item(r["norm_gt"], vocab, cfg["k"], cfg["seed"], r["Video file"])
            frames = extract_frames(os.path.join(cfg["video_dir"], r["Video file"]), cfg["n_frames"], cfg["max_side"])
            try:    raw = client.generate(frames, item["prompt"])
            except Exception as e: raw = f"__ERROR__ {e}"
            fh.write(json.dumps(dict(video=r["Video file"], gold=item["gold"], options=item["options"],
                                     raw=raw, cfg={k: v for k, v in cfg.items() if "key" not in k})) + "\n")
            fh.flush()
    return out

def score(jsonl_path, n_boot=2000, seed=0):
    rows = [json.loads(l) for l in open(jsonl_path)]
    correct, status = [], []
    for r in rows:
        lab, st = parse(r["raw"], set(r["options"]))
        status.append(st); correct.append(lab == r["gold"])
    correct = np.array(correct); ok = np.array([s == "ok" for s in status])
    k = len(rows[0]["options"]); rng = np.random.default_rng(seed)
    boots = [correct[rng.integers(0, len(rows), len(rows))].mean() for _ in range(n_boot)]
    return dict(n=len(rows), k=k, chance=1/k,
                accuracy_all=correct.mean(),
                accuracy_parseable=correct[ok].mean() if ok.any() else float("nan"),
                unparseable_rate=1 - ok.mean(),
                ci95=(np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
