"""
Dataset analysis script for ASL Citizen (stratified sample).
Computes volume, duration, video properties, text/gloss stats,
signer overlap, correlations, and data quality checks.
Outputs: a stats summary + PNG figures into outputs/analysis/
"""
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("/content/aslvlm/data")
VIDEO_DIR = DATA_DIR / "videos"
MANIFEST = Path("/content/aslvlm/data_sample_manifest.csv")
OUT_DIR = Path("/content/aslvlm/outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------- Load manifest ----------
df = pd.read_csv(MANIFEST)
print(f"Loaded manifest: {len(df)} rows")
print(df["split"].value_counts())

# ---------- 1. Volume per split ----------
volume = df.groupby("split").size().rename("num_clips")
print("\n=== Volume per split ===")
print(volume)

# ---------- 2. Video-level properties (duration, fps, resolution) ----------
records = []
missing = []
for _, row in df.iterrows():
    path = VIDEO_DIR / row["Video file"]
    if not path.exists():
        missing.append(row["Video file"])
        continue
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        missing.append(row["Video file"])
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = frame_count / fps if fps > 0 else np.nan
    cap.release()
    records.append({
        "Video file": row["Video file"],
        "split": row["split"],
        "Gloss": row["Gloss"],
        "Participant ID": row["Participant ID"],
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    })

video_stats = pd.DataFrame(records)
print(f"\nReadable videos: {len(video_stats)} | Unreadable/missing: {len(missing)}")
if missing:
    pd.Series(missing).to_csv(OUT_DIR / "unreadable_or_missing_videos.csv", index=False)

# Flag suspicious durations
zero_or_negative = video_stats[video_stats["duration_sec"] <= 0]
extreme_long = video_stats[video_stats["duration_sec"] > video_stats["duration_sec"].quantile(0.995)]
print(f"Zero/negative duration clips: {len(zero_or_negative)}")
print(f"Extreme-length outliers (>99.5th pct): {len(extreme_long)}")

# ---------- 3. Text / gloss stats ----------
gloss_counts = df["Gloss"].value_counts()
print("\n=== Gloss (vocab) stats ===")
print(f"Vocabulary size: {df['Gloss'].nunique()}")
print(f"Most frequent glosses:\n{gloss_counts.head(10)}")

train_vocab = set(df[df["split"] == "train"]["Gloss"].unique())
val_vocab = set(df[df["split"] == "val"]["Gloss"].unique())
test_vocab = set(df[df["split"] == "test"]["Gloss"].unique())
oov_val = val_vocab - train_vocab
oov_test = test_vocab - train_vocab
print(f"OOV rate (val vs train): {len(oov_val)/len(val_vocab)*100:.2f}%")
print(f"OOV rate (test vs train): {len(oov_test)/len(test_vocab)*100:.2f}%")

# ---------- 4. Signer overlap across splits ----------
train_signers = set(df[df["split"] == "train"]["Participant ID"].unique())
val_signers = set(df[df["split"] == "val"]["Participant ID"].unique())
test_signers = set(df[df["split"] == "test"]["Participant ID"].unique())

overlap_train_val = train_signers & val_signers
overlap_train_test = train_signers & test_signers
overlap_val_test = val_signers & test_signers

print("\n=== Signer overlap across splits ===")
print(f"Train signers: {len(train_signers)} | Val: {len(val_signers)} | Test: {len(test_signers)}")
print(f"Train∩Val: {len(overlap_train_val)} | Train∩Test: {len(overlap_train_test)} | Val∩Test: {len(overlap_val_test)}")
if overlap_train_test or overlap_train_val:
    print("WARNING: signer overlap detected across splits — evaluation may be signer-dependent, not signer-independent.")

clips_per_signer = df.groupby("Participant ID").size()

# ---------- 5. Duration vs sentence-length-equivalent (signing rate) ----------
# ASL Citizen is isolated-sign, so "sentence length" isn't applicable —
# instead check duration vs gloss-name length as a rough proxy, and flag outliers.
video_stats["gloss_len_chars"] = video_stats["Gloss"].str.len()

# ---------- FIGURES ----------

# Fig 1: Duration distribution
plt.figure(figsize=(7, 4))
sns.histplot(video_stats["duration_sec"], bins=50, color="#4C72B0")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("Clip duration distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_duration_distribution.png")
plt.close()

# Fig 2: FPS / resolution variation
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(video_stats["fps"].dropna(), bins=30, ax=axes[0], color="#55A868")
axes[0].set_title("FPS distribution")
axes[0].set_xlabel("FPS")
sns.scatterplot(x="width", y="height", data=video_stats, ax=axes[1], alpha=0.3, color="#C44E52")
axes[1].set_title("Resolution scatter (width vs height)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_fps_resolution.png")
plt.close()

# Fig 3: Clips per signer
plt.figure(figsize=(8, 4))
clips_per_signer.sort_values(ascending=False).plot(kind="bar", color="#8172B2")
plt.xlabel("Participant ID")
plt.ylabel("Number of clips (in sample)")
plt.title("Clips per signer")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_clips_per_signer.png")
plt.close()

# Fig 4: Volume per split
plt.figure(figsize=(5, 4))
volume.plot(kind="bar", color="#DD8452")
plt.ylabel("Number of clips")
plt.title("Sample volume per split")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_volume_per_split.png")
plt.close()

# ---------- Save summary CSVs ----------
video_stats.to_csv(OUT_DIR / "video_stats.csv", index=False)
gloss_counts.to_csv(OUT_DIR / "gloss_counts.csv")

print(f"\nAll outputs saved to {OUT_DIR}")
