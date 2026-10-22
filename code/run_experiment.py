"""
run_experiment.py — 產生報告所需的全部實驗數據與圖表
用法: python run_experiment.py <images_dir> <video_path> <out_dir>
"""
import sys, os, glob, json, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from autofocus import (METHODS, preprocess, score_frames, read_video_frames,
                       edge_maps, normalize)

# 中文字型
for f in ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"Sobel": "#2563eb", "Laplacian": "#dc2626", "Canny": "#16a34a"}

img_dir, video_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
summary = {}

# ============================================================
# 1. 十張靜態影像
# ============================================================
paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
names = [os.path.basename(p) for p in paths]
imgs = [cv2.imread(p) for p in paths]
img_scores, img_times = score_frames(imgs)

summary["images"] = {
    "names": names,
    "scores": {k: v.tolist() for k, v in img_scores.items()},
    "best": {k: names[int(np.argmax(v))] for k, v in img_scores.items()},
    "time_ms_per_img": {k: v / len(imgs) * 1000 for k, v in img_times.items()},
}

# 排名
for k, v in img_scores.items():
    order = np.argsort(-v)
    summary["images"].setdefault("rank", {})[k] = [names[i] for i in order]

# --- 圖 1:圖片清晰度曲線(正規化)
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(1, len(names) + 1)
for k, v in img_scores.items():
    ax.plot(x, normalize(v), "-o", color=COLORS[k], lw=2, label=k)
    b = int(np.argmax(v))
    ax.scatter([x[b]], [1.0], s=220, facecolors="none", edgecolors=COLORS[k], lw=2)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(".jpg", "") for n in names], rotation=30)
ax.set_ylabel("正規化清晰度分數 (0~1)")
ax.set_xlabel("影像檔名(對焦位置)")
ax.set_title("十張影像之清晰度分數比較")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{out}/fig_images_curve.png", dpi=180); plt.close(fig)

# --- 圖 2:原始分數長條圖(三個子圖)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (k, v) in zip(axes, img_scores.items()):
    bars = ax.bar(x, v, color=COLORS[k], alpha=.35)
    bars[int(np.argmax(v))].set_alpha(1.0)
    ax.set_title(f"{k}(最佳:{names[int(np.argmax(v))]})", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x])
    ax.grid(axis="y", alpha=.3)
fig.suptitle("各方法原始分數(深色為最佳影像)")
fig.tight_layout(); fig.savefig(f"{out}/fig_images_bars.png", dpi=180); plt.close(fig)

# --- 圖 3:邊緣偵測對照(最模糊 / 中間 / 最清晰)
lap = img_scores["Laplacian"]
pick = [int(np.argmin(lap)), int(np.argsort(lap)[len(lap) // 2]), int(np.argmax(lap))]
labels = ["最模糊", "中等", "最清晰"]
fig, axes = plt.subplots(3, 4, figsize=(14, 8.2))
for r, (idx, lab) in enumerate(zip(pick, labels)):
    g = preprocess(imgs[idx])
    em = edge_maps(g)
    axes[r, 0].imshow(cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB))
    axes[r, 0].set_title(f"{lab}:{names[idx]}", fontsize=10)
    for c, k in enumerate(["Sobel", "Laplacian", "Canny"], start=1):
        axes[r, c].imshow(em[k], cmap="gray")
        axes[r, c].set_title(f"{k}  分數={img_scores[k][idx]:.4g}", fontsize=10)
for a in axes.ravel(): a.axis("off")
fig.tight_layout(); fig.savefig(f"{out}/fig_edge_compare.png", dpi=150); plt.close(fig)

# ============================================================
# 2. 影片逐幀
# ============================================================
frames, fps = read_video_frames(video_path)
vid_scores, vid_times = score_frames(frames)
n = len(frames)
best_frames = {k: int(np.argmax(v)) for k, v in vid_scores.items()}
summary["video"] = {
    "n_frames": n, "fps": fps, "duration_s": n / fps,
    "best_frame": best_frames,
    "best_time_s": {k: v / fps for k, v in best_frames.items()},
    "time_ms_per_frame": {k: v / n * 1000 for k, v in vid_times.items()},
    "total_time_s": {k: v for k, v in vid_times.items()},
    "scores": {k: v.tolist() for k, v in vid_scores.items()},
}
for k, b in best_frames.items():
    cv2.imwrite(f"{out}/best_frame_{k}_{b:03d}.jpg", frames[b])

# --- 圖 4:影片清晰度曲線
fig, ax = plt.subplots(figsize=(11, 4.5))
t = np.arange(n)
for k, v in vid_scores.items():
    ax.plot(t, normalize(v), color=COLORS[k], lw=1.6, label=f"{k}(最佳幀 #{best_frames[k]})")
    ax.axvline(best_frames[k], color=COLORS[k], ls="--", lw=1, alpha=.7)
ax.set_xlabel(f"影格編號 (frame, {fps:.0f} fps)")
ax.set_ylabel("正規化清晰度分數")
ax.set_title(f"影片 AF.mp4 逐幀清晰度變化(共 {n} 幀)")
ax.grid(alpha=.3); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{out}/fig_video_curve.png", dpi=180); plt.close(fig)

# --- 圖 5:執行時間比較
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = list(METHODS)
vals = [summary["video"]["time_ms_per_frame"][k] for k in ks]
bars = ax.bar(ks, vals, color=[COLORS[k] for k in ks])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} ms", ha="center", va="bottom")
ax.set_ylabel("每幀平均運算時間 (ms)")
ax.set_title("各方法執行時間比較(1280×720)")
ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{out}/fig_time.png", dpi=180); plt.close(fig)

# --- 圖 6:最佳幀縮圖並排
fig, axes = plt.subplots(1, 3, figsize=(13, 2.9))
for ax, k in zip(axes, ks):
    ax.imshow(cv2.cvtColor(frames[best_frames[k]], cv2.COLOR_BGR2RGB))
    ax.set_title(f"{k} 選出:第 {best_frames[k]} 幀 ({best_frames[k]/fps:.2f}s)", fontsize=10)
    ax.axis("off")
fig.tight_layout(); fig.savefig(f"{out}/fig_best_frames.png", dpi=150); plt.close(fig)

# ============================================================
# 2b. 中心 ROI 改良(只評估畫面中央 50%)
# ============================================================
roi_scores, roi_times = score_frames(frames, roi_ratio=0.5)
roi_best = {k: int(np.argmax(v)) for k, v in roi_scores.items()}
summary["video_roi"] = {
    "roi_ratio": 0.5,
    "best_frame": roi_best,
    "best_time_s": {k: v / fps for k, v in roi_best.items()},
    "time_ms_per_frame": {k: v / n * 1000 for k, v in roi_times.items()},
    "scores": {k: v.tolist() for k, v in roi_scores.items()},
}
for k, b in roi_best.items():
    cv2.imwrite(f"{out}/best_frame_ROI_{k}_{b:03d}.jpg", frames[b])

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (sc, bf, ttl) in zip(axes, [(vid_scores, best_frames, "全畫面"),
                                    (roi_scores, roi_best, "中心 ROI 50%")]):
    for k, v in sc.items():
        ax.plot(t, normalize(v), color=COLORS[k], lw=1.5, label=f"{k} → #{bf[k]}")
        ax.axvline(bf[k], color=COLORS[k], ls="--", lw=1, alpha=.7)
    ax.set_xlim(70, 160); ax.set_title(ttl); ax.set_xlabel("影格編號")
    ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper right")
axes[0].set_ylabel("正規化清晰度分數")
fig.suptitle("全畫面 vs 中心 ROI:最佳焦點選擇差異(放大 70~160 幀)")
fig.tight_layout(); fig.savefig(f"{out}/fig_roi_compare.png", dpi=180); plt.close(fig)

# 92 vs 102 幀局部放大對照(前景文字 vs 背景紋理)
crops = [(frames[best_frames["Sobel"]], f"第 {best_frames['Sobel']} 幀"),
         (frames[best_frames["Laplacian"]], f"第 {best_frames['Laplacian']} 幀")]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
for c, (fr, lab) in enumerate(crops):
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    axes[0, c].imshow(rgb[330:470, 460:780]); axes[0, c].set_title(f"{lab}|前景文字", fontsize=10)
    axes[1, c].imshow(rgb[420:600, 900:1220]); axes[1, c].set_title(f"{lab}|背景牆面", fontsize=10)
for a in axes.ravel(): a.axis("off")
fig.tight_layout(); fig.savefig(f"{out}/fig_fg_bg_zoom.png", dpi=150); plt.close(fig)

# ============================================================
# 3. 雜訊敏感度測試
# ============================================================
base = imgs[int(np.argmax(lap))]
sigmas = [0, 2, 5, 10, 15, 20, 30]
noise_res = {k: [] for k in METHODS}
rng = np.random.default_rng(0)
for s in sigmas:
    noisy = np.clip(base.astype(float) + rng.normal(0, s, base.shape), 0, 255).astype(np.uint8)
    g = preprocess(noisy)
    for k, fn in METHODS.items():
        noise_res[k].append(fn(g))
# 同時加雜訊到最模糊的影像,看是否被誤判為比清晰影像還清楚
blur = imgs[int(np.argmin(lap))]
noise_blur = {k: [] for k in METHODS}
rng = np.random.default_rng(0)
for s in sigmas:
    noisy = np.clip(blur.astype(float) + rng.normal(0, s, blur.shape), 0, 255).astype(np.uint8)
    g = preprocess(noisy)
    for k, fn in METHODS.items():
        noise_blur[k].append(fn(g))
summary["noise"] = {"sigmas": sigmas, "sharp": noise_res, "blur": noise_blur}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, k in zip(axes, ks):
    ax.plot(sigmas, noise_res[k], "-o", color=COLORS[k], label="清晰影像 + 雜訊")
    ax.plot(sigmas, noise_blur[k], "--s", color="gray", label="模糊影像 + 雜訊")
    ax.set_title(k); ax.set_xlabel("高斯雜訊 σ"); ax.grid(alpha=.3)
axes[0].set_ylabel("清晰度分數"); axes[0].legend(fontsize=8)
fig.suptitle("雜訊敏感度測試:分數是否被雜訊灌水")
fig.tight_layout(); fig.savefig(f"{out}/fig_noise.png", dpi=180); plt.close(fig)

# ============================================================
# 4. 方法一致性 (Spearman rank correlation)
# ============================================================
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
summary["consistency_video"] = {
    "Sobel-Laplacian": spearman(vid_scores["Sobel"], vid_scores["Laplacian"]),
    "Sobel-Canny": spearman(vid_scores["Sobel"], vid_scores["Canny"]),
    "Laplacian-Canny": spearman(vid_scores["Laplacian"], vid_scores["Canny"]),
}

with open(f"{out}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)

# 終端輸出
print("== 圖片 ==")
print(f"{'file':<12}" + "".join(f"{k:>14}" for k in METHODS))
for i, nm in enumerate(names):
    print(f"{nm:<12}" + "".join(f"{img_scores[k][i]:>14.4f}" for k in METHODS))
print("best:", summary["images"]["best"])
print("ms/img:", {k: round(v, 2) for k, v in summary["images"]["time_ms_per_img"].items()})
print("\n== 影片 ==", n, "frames @", fps)
print("best frame:", best_frames)
print("best time:", {k: round(v, 3) for k, v in summary["video"]["best_time_s"].items()})
print("ms/frame:", {k: round(v, 2) for k, v in summary["video"]["time_ms_per_frame"].items()})
print("spearman:", summary["consistency_video"])
print("ROI best:", roi_best, {k: round(v, 2) for k, v in summary["video_roi"]["time_ms_per_frame"].items()})
print("\n== 雜訊 ==")
for k in METHODS:
    print(k, "sharp:", [round(x, 3) for x in noise_res[k]])
    print(k, "blur :", [round(x, 3) for x in noise_blur[k]])
