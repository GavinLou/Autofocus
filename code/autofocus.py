"""
autofocus.py — 以邊緣資訊評估影像清晰度的核心模組

提供三種清晰度 (Focus Measure) 計算方法:
  1. Sobel  (Tenengrad)      : 梯度能量平均值
  2. Laplacian (Variance)    : 拉普拉斯響應的變異數
  3. Canny  (Edge Density)   : 邊緣像素所佔比例

作者:劉冠廷 (NYUST CSIE)
"""
import time
import cv2
import numpy as np

# ------------------------------------------------------------
# 前處理
# ------------------------------------------------------------
def preprocess(img_bgr, roi_ratio=1.0, max_width=None):
    """轉灰階、(可選)中心 ROI 裁切、(可選)縮小。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    if roi_ratio < 1.0:
        h, w = gray.shape
        rh, rw = int(h * roi_ratio), int(w * roi_ratio)
        y0, x0 = (h - rh) // 2, (w - rw) // 2
        gray = gray[y0:y0 + rh, x0:x0 + rw]
    if max_width and gray.shape[1] > max_width:
        s = max_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray


# ------------------------------------------------------------
# 三種清晰度評估方法
# ------------------------------------------------------------
def sobel_tenengrad(gray):
    """Tenengrad: mean(Gx^2 + Gy^2)"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_variance(gray):
    """Variance of Laplacian: Var(∇²I)"""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def canny_density(gray, low=50, high=150):
    """Edge density: (# edge pixels) / (# total pixels)"""
    edges = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(edges)) / edges.size


METHODS = {
    "Sobel": sobel_tenengrad,
    "Laplacian": laplacian_variance,
    "Canny": canny_density,
}


# ------------------------------------------------------------
# 邊緣圖(視覺化用)
# ------------------------------------------------------------
def edge_maps(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sob = cv2.convertScaleAbs(np.sqrt(gx ** 2 + gy ** 2))
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    can = cv2.Canny(gray, 50, 150)
    return {"Sobel": sob, "Laplacian": lap, "Canny": can}


# ------------------------------------------------------------
# 評分整批影像 / 影片
# ------------------------------------------------------------
def score_frames(frames, **pre_kw):
    """frames: list of BGR images -> dict(method -> np.array scores), dict(method -> total sec)"""
    grays = [preprocess(f, **pre_kw) for f in frames]
    scores, times = {}, {}
    for name, fn in METHODS.items():
        fn(grays[0])  # 暖機,避免第一次呼叫的初始化時間影響計時
        t0 = time.perf_counter()
        scores[name] = np.array([fn(g) for g in grays])
        times[name] = time.perf_counter() - t0
    return scores, times


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def normalize(arr):
    arr = np.asarray(arr, dtype=float)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)
