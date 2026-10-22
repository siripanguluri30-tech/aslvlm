# asl_ext.py -- config-driven runs, frame sampling strategies, frame cache, stratified scoring.
# Needs /content/asl_quickstart.py (as module) to exist.
import sys; sys.path.insert(0, "/content")
import os, json, hashlib, itertools, random
import numpy as np, pandas as pd, cv2, yaml
from PIL import Image
import asl_quickstart as Q

CACHE = "/content/cache/frames"
RUNS = "/content/outputs/runs"

# ---------- frame sampling (Task 1.3) + cache (Task 4.2) ----------
def _read_gray_small(path):
    cap = cv2.VideoCapture(path); g = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.float32))
    cap.release(); return g

def pick_indices(path, strategy, n):
    cap = cv2.VideoCapture(path); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    T = max(T, 1)
    if strategy == "uniform":                       # full clip, endpoints included (what the first runs used)
        idx = np.linspace(0, T - 1, n) if n > 1 else np.array([(T - 1) / 2])
    elif strategy == "middle60":                    # trim the first/last 20%
        idx = np.linspace(0.2 * (T - 1), 0.8 * (T - 1), n) if n > 1 else np.array([(T - 1) / 2])
    elif strategy == "motion":                      # dense where inter-frame difference is highest
        g = _read_gray_small(path); T = len(g)
        if T < 2: return np.zeros(n, int)
        m = np.array([np.abs(g[i] - g[i - 1]).mean() for i in range(1, T)])
        w = m + 0.1 * m.mean() + 1e-6               # small floor so still-but-informative frames are not ignored
        cdf = np.cumsum(w) / w.sum()
        q = (np.arange(n) + 0.5) / n                # equal slices of the motion mass
        idx = np.searchsorted(cdf, q) + 1
        idx = np.clip(idx, 0, T - 1)
        used, fixed = set(), []                     # avoid picking the same frame twice
        for i in idx:
            j = 0
            while True:
                for c in (i - j, i + j):
                    if 0 <= c < T and c not in used: used.add(c); fixed.append(c); break
                else: j += 1; continue
                break
        idx = np.array(sorted(fixed))
    else:
        raise ValueError(strategy)
    return np.clip(np.round(idx).astype(int), 0, T - 1)

def get_frames(video_path, strategy, n, max_side):
    vid = os.path.splitext(os.path.basename(video_path))[0]
    d = f"{CACHE}/{vid}/{strategy}_{n}_{max_side}"
    if os.path.isdir(d) and len(os.listdir(d)) == n:
        return [Image.open(f"{d}/{i}.jpg").convert("RGB") for i in range(n)]
    idx = pick_indices(video_path, strategy, n)
    cap = cv2.VideoCapture(video_path); frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok: f = np.zeros((120, 160, 3), np.uint8)
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); im.thumbnail((max_side, max_side)); frames.append(im)
    cap.release()
    os.makedirs(d, exist_ok=True)
    for i, im in enumerate(frames): im.save(f"{d}/{i}.jpg", quality=92)
    return frames

# ---------- config-driven run (Task 4.1) ----------
DEFAULTS = dict(model="mock", frames=dict(n=6, strategy="uniform", max_side=384),
                task=dict(type="forced_choice", n_candidates=4, distractors="random"),
                split="/content/aslvlm/splits/test_signer_independent.txt", n_clips=200, seed=0)
ROOT = dict(manifest="/content/aslvlm/data/data_sample_manifest.csv", video_dir="/content/aslvlm/data/ASL_Citizen_sample/videos")

def resolve(cfg):
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in cfg.items():
        if isinstance(v, dict): out[k].update(v)
        else: out[k] = v
    return out

def run_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def run_config(cfg, client, max_errors=3):
    cfg = resolve(cfg); rid = run_id(cfg)
    d = f"{RUNS}/{cfg['model']}_f{cfg['frames']['n']}_{cfg['frames']['strategy']}_k{cfg['task']['n_candidates']}_{rid}"
    os.makedirs(d, exist_ok=True)
    json.dump(cfg, open(f"{d}/config.resolved.json", "w"), indent=1)       # config next to predictions
    pred = f"{d}/predictions.jsonl"
    if cfg["task"]["distractors"] != "random": raise NotImplementedError("hard distractors need ASL-LEX data")
    m = pd.read_csv(ROOT["manifest"]); m["norm_gt"] = m["Gloss"].str.lower().str.strip()
    ids = set(open(cfg["split"]).read().splitlines())
    df = m[m["Video file"].isin(ids)].sample(n=min(cfg["n_clips"], len(ids)), random_state=cfg["seed"]).reset_index(drop=True)
    vocab = sorted(m["norm_gt"].unique())
    done = {json.loads(l)["video"] for l in open(pred)} if os.path.exists(pred) else set()
    bad = 0
    with open(pred, "a") as fh:
        for _, r in df.iterrows():
            if r["Video file"] in done: continue
            item = Q.build_item(r["norm_gt"], vocab, cfg["task"]["n_candidates"], cfg["seed"], r["Video file"])
            fr = get_frames(os.path.join(ROOT["video_dir"], r["Video file"]), cfg["frames"]["strategy"], cfg["frames"]["n"], cfg["frames"]["max_side"])
            try: raw = client.generate(fr, item["prompt"])
            except Exception as e: raw = f"__ERROR__ {e}"
            if raw.startswith("__ERROR__"):
                bad += 1; print("error:", raw[:150])
                if bad >= max_errors: print("Stopping; rerun later, finished clips are kept."); break
                continue
            bad = 0
            fh.write(json.dumps(dict(video=r["Video file"], signer=r["Participant ID"], gold=item["gold"],
                                     options=item["options"], raw=raw)) + "\n"); fh.flush()
    return pred

def load_cfg(path):
    return yaml.safe_load(open(path))

def run_grid(base, client, frames_n=(6,), strategies=("uniform",), ks=(4,)):
    paths = []
    for n, s, k in itertools.product(frames_n, strategies, ks):
        c = json.loads(json.dumps(base)); c.setdefault("frames", {}).update(n=n, strategy=s)
        c.setdefault("task", {}).update(n_candidates=k)
        print(f"--- frames={n} strategy={s} k={k}")
        paths.append(run_config(c, client))
    return paths

# ---------- stratified scoring (Task 3.3, 5.3) ----------
def _meta():
    m = pd.read_csv(ROOT["manifest"]); m["norm_gt"] = m["Gloss"].str.lower().str.strip()
    return m, m["norm_gt"].value_counts()

def score_run(pred_path, n_boot=2000, seed=0):
    rows = [json.loads(l) for l in open(pred_path)]
    m, gcount = _meta(); by_vid = m.set_index("Video file")
    recs = []
    for r in rows:
        lab, st = Q.parse(r["raw"], set(r["options"]))
        v = by_vid.loc[r["video"]]
        cap = cv2.VideoCapture(os.path.join(ROOT["video_dir"], r["video"]))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1); cap.release()
        recs.append(dict(video=r["video"], signer=v["Participant ID"], correct=(lab == r["gold"]), status=st,
                         gloss_freq=int(gcount[v["norm_gt"]]), duration=dur, k=len(r["options"])))
    d = pd.DataFrame(recs); rng = np.random.default_rng(seed)
    c = d["correct"].values.astype(float)
    boot = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)]
    sig = d["signer"].unique(); groups = {s: d.index[d.signer == s].values for s in sig}
    cb = []
    for _ in range(n_boot):
        pick = rng.choice(sig, len(sig)); ix = np.concatenate([groups[s] for s in pick]); cb.append(c[ix].mean())
    d["freq_group"] = np.where(d.gloss_freq >= 3, "frequent(>=3)", "rare(<3)")
    d["dur_q"] = ("Q" + (pd.qcut(d.duration.rank(method="first"), 4, labels=False) + 1).astype(str))   # Q1 = shortest, Q4 = longest
    summ = dict(n=len(d), k=int(d.k.iloc[0]), chance=1 / int(d.k.iloc[0]), acc=c.mean(),
                ci_clip=tuple(np.percentile(boot, [2.5, 97.5])), ci_signer=tuple(np.percentile(cb, [2.5, 97.5])),
                unparseable=float((d.status != "ok").mean()))
    br = {name: d.groupby(col, observed=True)["correct"].agg(["count", "mean"]).round(3)
          for name, col in [("by_signer", "signer"), ("by_gloss_freq", "freq_group"), ("by_duration_quartile", "dur_q")]}
    return summ, br

def table(paths):
    out = []
    for p in paths:
        cfg = json.load(open(os.path.join(os.path.dirname(p), "config.resolved.json"))); s, _ = score_run(p)
        out.append(dict(model=cfg["model"], frames=cfg["frames"]["n"], strategy=cfg["frames"]["strategy"], k=s["k"], n=s["n"],
                        acc=round(s["acc"], 3), ci_clip=tuple(round(x, 3) for x in s["ci_clip"]),
                        ci_signer=tuple(round(x, 3) for x in s["ci_signer"]), chance=s["chance"], unparseable=round(s["unparseable"], 3)))
    return pd.DataFrame(out)
