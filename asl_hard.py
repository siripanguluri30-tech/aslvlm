# asl_hard.py -- hard distractors from ASL-LEX 2.0 (Task 2.2).
# A hard distractor shares the answer's handshape and major location but differs in movement.
# Needs /content/signdata.csv (ASL-LEX 2.0 data table, latin-1) and the manifest's "ASL-LEX Code" column.
import sys; sys.path.insert(0, "/content")
import re, random
import pandas as pd
import asl_quickstart as Q

LEX_PATH = "/content/signdata.csv"
MANIFEST = "/content/aslvlm/data/data_sample_manifest.csv"
TIERS = ["T1 same handshape+location, different movement",
         "T2 same handshape, different movement",
         "T3 same location, different movement",
         "T4 same handshape or location (movement same or unknown)",
         "T5 anything else (random)"]
_FEATS = None

def _clean(x):
    return None if pd.isna(x) else str(x)

def load_feats(manifest=None, lex_path=None):
    """gloss (lowercased, as in the manifest) -> dict(code, hs, loc, mov). Uses the first morpheme's coding."""
    global _FEATS
    lex = pd.read_csv(lex_path or LEX_PATH, encoding="latin-1").drop_duplicates("Code").set_index("Code")
    m = pd.read_csv(manifest or MANIFEST); m["g"] = m["Gloss"].str.lower().str.strip()
    codes = m.dropna(subset=["ASL-LEX Code"]).groupby("g")["ASL-LEX Code"].first()
    feats = {}
    for g, c in codes.items():
        if c in lex.index:
            r = lex.loc[c]
            feats[g] = dict(code=c, hs=_clean(r["Handshape.2.0"]), loc=_clean(r["MajorLocation.2.0"]), mov=_clean(r["Movement.2.0"]))
    _FEATS = feats
    return feats

def _tier(a, b):
    """1..4 for a candidate b against the answer a; 5 = unrelated."""
    hs = a["hs"] is not None and a["hs"] == b["hs"]
    loc = a["loc"] is not None and a["loc"] == b["loc"]
    mov_diff = a["mov"] is not None and b["mov"] is not None and a["mov"] != b["mov"]
    if hs and loc and mov_diff: return 1
    if hs and mov_diff: return 2
    if loc and mov_diff: return 3
    if hs or loc: return 4
    return 5

def tier_pools(correct, vocab, feats=None):
    feats = feats or _FEATS or load_feats()
    a = feats.get(correct)
    base = re.sub(r"\d+$", "", correct)
    pools = {t: [] for t in range(1, 6)}
    for v in vocab:
        if v == correct or re.sub(r"\d+$", "", v) == base: continue   # the answer itself and its numbered variants
        b = feats.get(v)
        if a is None or b is None or b["code"] == a["code"]: pools[5].append(v)
        else: pools[_tier(a, b)].append(v)
    return pools, a is not None

def build_item_hard(correct, vocab, k, seed, video_id):
    """Same prompt and return shape as Q.build_item; distractors drawn from the hardest tiers first."""
    rng = random.Random(f"{seed}-{video_id}-{k}-hard")
    pools, has = tier_pools(correct, vocab)
    chosen, used = [], []
    for t in range(1, 6):
        need = (k - 1) - len(chosen)
        if need <= 0: used.append(0); continue
        take = rng.sample(pools[t], min(need, len(pools[t])))
        chosen += take; used.append(len(take))
    opts = chosen + [correct]; rng.shuffle(opts)
    labels = Q.make_labels(k)
    body = "\n".join(f"{l}. {o}" for l, o in zip(labels, opts))
    prompt = (f"These frames are from a video of one American Sign Language sign.\n"
              f"Which candidate is being signed?\n\n{body}\n\n"
              f"Reply with only the label ({labels[0]}-{labels[-1]}).")
    return dict(prompt=prompt, gold=labels[opts.index(correct)], options=dict(zip(labels, opts)),
                meta=dict(answer_has_features=has, tier_counts=used))

def pool_stats(glosses, vocab, ks=(4, 10, 50)):
    """How hard are the hard distractors? For each k: share of questions whose distractors all come from T1 / T1-T3,
    mean number of distractors from each tier, and share of answers with no ASL-LEX features (random fallback)."""
    feats = _FEATS or load_feats()
    rows = []
    sizes = []
    for g in glosses:
        pools, has = tier_pools(g, vocab, feats)
        sizes.append((has, [len(pools[t]) for t in range(1, 6)]))
    for k in ks:
        n_all1 = n_123 = 0; cnt = [0.0] * 5; nf = 0
        for has, s in sizes:
            need = k - 1; got = []
            for t in range(5):
                x = min(need, s[t]); got.append(x); need -= x
            cnt = [c + x for c, x in zip(cnt, got)]
            if not has: nf += 1
            if has and got[0] == k - 1: n_all1 += 1
            if has and sum(got[:3]) == k - 1: n_123 += 1
        n = len(sizes)
        rows.append(dict(k=k, answers=n, all_T1=round(n_all1 / n, 3), all_T1_to_T3=round(n_123 / n, 3),
                         no_features=round(nf / n, 3), **{f"mean_T{t+1}": round(cnt[t] / n, 2) for t in range(5)}))
    return pd.DataFrame(rows)
