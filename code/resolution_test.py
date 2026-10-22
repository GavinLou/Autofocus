"""
resolution_test.py — 分析解析度對最佳焦點判斷的影響
用法: python resolution_test.py <video_path> <out_json>
"""
import sys, json
import numpy as np
from autofocus import read_video_frames, score_frames

frames, fps = read_video_frames(sys.argv[1])
res = {}
for mw in [1280, 640, 320]:
    s, t = score_frames(frames, max_width=mw)
    res[mw] = {"best": {k: int(np.argmax(v)) for k, v in s.items()},
               "ms_per_frame": {k: round(v / len(frames) * 1000, 2) for k, v in t.items()}}
    print(mw, res[mw])
json.dump(res, open(sys.argv[2], "w"), indent=1)
