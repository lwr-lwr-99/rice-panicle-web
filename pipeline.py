# -*- coding: utf-8 -*-
"""
穗长测量管线：YOLOv11n-Pose 关键点检测 + 主穗轴平滑弧线提取。
与 Streamlit 界面解耦，便于单独测试与复用。
"""
import os
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from skimage.morphology import skeletonize

DEFAULT_SCALE = 0.009141  # cm/px：5cm ≈ 547px；机位变化需重新标定


def euclidean_px(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def segment_panicle(crop):
    """黑背景下灰度分割稻穗，取最大连通域。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == big, 255, 0).astype(np.uint8)
    return mask


def skeletonize_mask(mask):
    skel = skeletonize(mask > 0)
    return (skel * 255).astype(np.uint8)


def main_axis_from_skel(skel):
    """骨架上最长路径（树的直径，两次 BFS）= 主穗轴。"""
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 2:
        return 0.0, []

    def bfs(start):
        dist = {start: 0.0}
        prev = {start: None}
        q = deque([start])
        far, far_d = start, 0.0
        while q:
            cx, cy = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb not in pts or nb in dist:
                        continue
                    w = 1.41421356 if dx != 0 and dy != 0 else 1.0
                    dist[nb] = dist[(cx, cy)] + w
                    prev[nb] = (cx, cy)
                    q.append(nb)
                    if dist[nb] > far_d:
                        far_d, far = dist[nb], nb
        return far, far_d, prev

    A, _, _ = bfs(next(iter(pts)))
    B, max_len, prev = bfs(A)
    path = []
    cur = B
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return max_len, path


def trim_path_to_kpts(path, p_base, p_tip):
    """把主穗轴路径裁剪到穗基~穗尖之间，剔除节以下裸茎秆。"""
    if not path:
        return path

    def nearest_idx(pt):
        best_i, best_d = 0, float("inf")
        for i, (px, py) in enumerate(path):
            d = (px - pt[0]) ** 2 + (py - pt[1]) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    i0 = nearest_idx(p_base)
    i1 = nearest_idx(p_tip)
    if i0 > i1:
        i0, i1 = i1, i0
    return path[i0:i1 + 1]


def path_len(path):
    """路径真实欧氏长度（像素）。"""
    total = 0.0
    for i in range(len(path) - 1):
        total += float(np.hypot(path[i + 1][0] - path[i][0],
                                path[i + 1][1] - path[i][1]))
    return total


def _rolling_mean(a, w):
    """等权滑动平均（边缘镜像补边）。"""
    pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    ker = np.ones(w) / w
    return np.convolve(ap, ker, mode="valid")


def _skel_shortest_path(skel, base, tip):
    """骨架上穗基~穗尖的最短路径：沿中央主穗轴连通，不绕侧枝梗（对比最长直径路径）。"""
    ys, xs = np.nonzero(skel)
    if len(xs) < 2:
        return []
    pts = list(zip(xs.tolist(), ys.tolist()))
    pset = set(pts)

    def nearest(pt):
        return min(pts, key=lambda p: (p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2)

    A, B = nearest(base), nearest(tip)
    dist = {A: 0.0}
    prev = {A: None}
    q = deque([A])
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == B:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cx + dx, cy + dy)
                if nb not in pset or nb in dist:
                    continue
                w = 1.41421356 if dx != 0 and dy != 0 else 1.0
                dist[nb] = dist[(cx, cy)] + w
                prev[nb] = (cx, cy)
                q.append(nb)
    if B not in prev:
        return []
    path, cur = [], B
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _cluster_walk(mask, base, tip, y_top, y_bot):
    """回退方案：逐行分簇，从穗基向上按连续性追踪最近簇中心。"""
    ys = np.arange(y_top, y_bot + 1)
    clusters = []
    for y in ys:
        xs = np.nonzero(mask[y])[0]
        if len(xs):
            segs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
            clusters.append([(float(s.mean()), len(s)) for s in segs])
        else:
            clusters.append([])
    x_walk = np.full(len(ys), np.nan)
    x_prev = float(base[0])
    for k in range(len(ys) - 1, -1, -1):
        cl = clusters[k]
        if cl:
            cx, _ = min(cl, key=lambda c: (abs(c[0] - x_prev), -c[1]))
            x_walk[k] = cx
            x_prev = cx
        else:
            x_walk[k] = x_prev
    valid = ~np.isnan(x_walk)
    if valid.sum() < 2:
        return ys, None
    x_walk = np.interp(ys, ys[valid], x_walk[valid])
    return ys, x_walk


def spine_from_mask(mask, base, tip):
    """主穗轴平滑弧线：逐行取稻穗实际中心（同一条穗的各段谷粒整体中点），
    再用稳健低阶多项式（4 次，按 MAD 迭代剔除跳点）拟合，得到圆滑且贴合穗轴弯曲的弧线；
    首尾各 18% 用 smoothstep 平滑过渡到穗尖 / 穗颈节，避免折角。"""
    H, W = mask.shape[:2]
    y_top = int(round(tip[1]))
    y_bot = int(round(base[1]))
    y_top = max(0, min(H - 1, y_top))
    y_bot = max(0, min(H - 1, y_bot))
    if y_bot < y_top:
        y_top, y_bot = y_bot, y_top

    ys = np.arange(y_top, y_bot + 1)
    n = len(ys)
    if n < 6:
        return []

    # 骨架路径只作为「哪一段属于该穗」的参考，不再直接决定 x
    skel = skeletonize_mask(mask)
    path = _skel_shortest_path(skel, base, tip)
    row_ref = {}
    for x, y in path:
        row_ref.setdefault(int(y), []).append(x)

    xs = np.full(n, np.nan)
    for k, y in enumerate(ys):
        idx = np.nonzero(mask[y])[0]
        if not len(idx):
            continue
        segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        ref = (float(np.median(row_ref[y])) if y in row_ref
               else (float(tip[0]) if k < n // 2 else float(base[0])))
        # 只保留靠近该穗穗轴的分段，排除画面里蹭进来的邻穗
        keep = [s for s in segs if abs((s.min() + s.max()) / 2.0 - ref) < 0.35 * W]
        if not keep:
            keep = segs
        xs[k] = (min(int(s.min()) for s in keep) + max(int(s.max()) for s in keep)) / 2.0

    valid = ~np.isnan(xs)
    if valid.sum() < 4:
        return []
    if valid.sum() < n:
        xs = np.interp(ys, ys[valid], xs[valid])

    # 圆滑化：对逐行中心做稳健低阶多项式拟合
    # （迭代剔除芒须/邻穗/支梗造成的跳点），得到圆润的主穗轴弧线，
    # 既不会像窗口滤波那样残留锯齿，也不会把整体弯曲拉直
    yy = ys.astype(float)
    deg = 4 if n >= 60 else max(1, min(4, n // 12))
    inl = np.ones(n, dtype=bool)
    coef = np.polyfit(yy, xs, deg)
    for _ in range(4):
        res = xs - np.polyval(coef, yy)
        med = float(np.median(res[inl])) if inl.any() else 0.0
        sigma = 1.4826 * float(np.median(np.abs(res[inl] - med))) if inl.any() else 1.0
        new_inl = np.abs(res - med) <= max(2.0, 3.0 * sigma)
        if new_inl.sum() < max(deg + 3, 10) or np.array_equal(new_inl, inl):
            break
        inl = new_inl
        coef = np.polyfit(yy[inl], xs[inl], deg)

    y_new = np.linspace(float(tip[1]), float(base[1]), 160)
    x_new = np.polyval(coef, y_new)

    # 端点过渡对齐：首尾各 18% 长度内用平滑权重（smoothstep）拉回到穗尖 / 穗颈节，
    # 权重两端导数为 0，不会在过渡区末端留下折角
    m = max(4, int(len(x_new) * 0.18))
    t = np.linspace(0.0, 1.0, m)
    w = 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)          # 1 → 0 的平滑衰减
    x_new[:m] += (float(tip[0]) - x_new[0]) * w
    x_new[-m:] += (float(base[0]) - x_new[-1]) * w[::-1]
    x_new[0], x_new[-1] = float(tip[0]), float(base[0])

    return [(float(x), float(y)) for x, y in zip(x_new, y_new)]


def find_grain_top(mask, path, tip_raw):
    """从空须顶端往下找米粒开始的拐点（局部宽度突增）。"""
    h, w = mask.shape[:2]
    top_y = int(tip_raw[1])
    pts = sorted([p for p in path if p[1] >= top_y], key=lambda p: p[1])
    if not pts:
        return tip_raw

    def local_w(px, py):
        x0 = max(0, int(px) - 15)
        x1 = min(w, int(px) + 16)
        return int((mask[py, x0:x1] > 0).sum())

    if local_w(tip_raw[0], top_y) > 8:
        return tip_raw
    for px, py in pts:
        py = int(py)
        if py >= h:
            continue
        if local_w(px, py) > 8:
            return (float(px), float(py))
    return tip_raw


def _snap_base_x(mask, gray_crop, y, fallback_x, search=40):
    """在穗基所在行（含上下邻近行）把 x 对齐到最靠近模型基点的茎秆/穗体簇中心，
    避免吸附到旁边更宽的谷粒簇而偏离茎秆中心线。"""
    H = mask.shape[0]
    rows = [y]
    for d in range(1, search + 1):
        if y - d >= 0:
            rows.append(y - d)
        if y + d < H:
            rows.append(y + d)
    for ry in rows:
        xs = np.nonzero(mask[ry])[0]
        if len(xs):
            segs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
            seg = min(segs, key=lambda s: abs(s.mean() - fallback_x))
            return float(seg.mean()), ry
    return float(fallback_x), y


def refine_keypoints(mask, crop_img, path, pred_base, pred_tip):
    """穗尖沿用骨架米粒拐点校正；穗基以模型原始关键点为准、只做横向中心吸附，
    不再上推到颗粒下缘（实测旧后处理会把穗基抬高、偏离人工标注的穗颈节）。"""
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # ---- 穗尖：骨架米粒拐点校正 ----
    if len(path) >= 2:
        tip_raw = min(path, key=lambda p: p[1])
        tip_y = find_grain_top(mask, path, tip_raw)
        ty = float(tip_y[1])
        tip_path_pt = min(path, key=lambda p: abs(p[1] - ty))
        tip = (float(tip_path_pt[0]), ty)
    else:
        tip = (float(pred_tip[0]), float(pred_tip[1]))

    # ---- 穗基：保留模型原始 y，x 吸附到茎秆/穗体中心 ----
    H = mask.shape[0]
    by = int(round(pred_base[1]))
    by = max(0, min(H - 1, by))
    x_base, by_used = _snap_base_x(mask, gray_crop, by, float(pred_base[0]))
    base = (x_base, float(by_used))
    return base, tip


# ----------------------------------------------------------------------
# 可视化：尺寸随图像分辨率自适应
# 4K 原图上固定 3px 线宽 / 10px 圆点，缩放到网页宽度后不足 1 像素，
# 肉眼几乎看不到——这是「关键点和黄线像没检测到」的根因。
# ----------------------------------------------------------------------
_APP_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT_CANDIDATES = [
    os.path.join(_APP_HERE, "simsun.ttc"),
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]

C_BASE_RGB = (95, 155, 255)    # 穗颈节（蓝）
C_TIP_RGB = (110, 235, 120)    # 穗尖（绿）
C_SPINE_RGB = (255, 226, 70)   # 主穗轴（黄）
C_LEN_RGB = (255, 236, 130)
C_LONGEST_RGB = (255, 95, 95)


@lru_cache(maxsize=64)
def _cjk_font(size):
    """加载中文字体；找不到时回退 PIL 默认字体。"""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _pil_texts(bgr, texts):
    """用 PIL 在 OpenCV 图上写字（cv2.putText 不支持中文）。
    texts: [(x, y, text, rgb, size, stroke_width)]"""
    if not texts:
        return bgr
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for x, y, text, rgb, size, stroke in texts:
        draw.text((x, y), text, font=_cjk_font(int(size)), fill=rgb,
                  stroke_width=int(stroke), stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _dashed_line(img, p1, p2, color, thickness):
    """虚线（主穗轴提取失效时，用来连接穗颈节与穗尖）。"""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    L = float(np.hypot(*(p2 - p1)))
    if L < 1:
        return
    seg = max(16.0, L / 30.0)
    k = 0.0
    while k < L:
        a = p1 + (p2 - p1) * (k / L)
        b = p1 + (p2 - p1) * (min(k + seg * 0.6, L) / L)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thickness, cv2.LINE_AA)
        k += seg


def _draw_simple(vis, items, marker_scale=1.0):
    """极简出图样式（贴近论文 / 比赛常用图）：
    蓝色检测框 + 细黄主穗轴曲线 + 蓝点穗颈节 + 绿点穗尖 + 左上角蓝色长度文字。"""
    H, W = vis.shape[:2]
    s = max(H, W) / 1000.0
    lw = max(2, int(round(1.3 * s)))
    r = max(3, int(round(3.6 * s * max(0.6, marker_scale))))
    fs = max(12, int(round(11.0 * s * max(0.8, marker_scale))))
    texts = []

    for it in items:
        x1, y1, x2, y2 = it["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), lw, cv2.LINE_AA)

        base, tip = it["base"], it["tip"]
        path = it.get("path") or []
        if len(path) >= 2:
            pts = np.array([[int(p[0]), int(p[1])] for p in path], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), lw, cv2.LINE_AA)
        else:
            cv2.line(vis, (int(base[0]), int(base[1])),
                     (int(tip[0]), int(tip[1])), (0, 255, 255), lw, cv2.LINE_AA)

        cv2.circle(vis, (int(base[0]), int(base[1])), r, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(tip[0]), int(tip[1])), r, (0, 255, 0), -1, cv2.LINE_AA)

        # 长度文字贴在检测框左上角外侧（贴顶时放进框内），与关键点圆点冲突则换位
        label = it.get("len_text", "")
        tw = float(_cjk_font(int(fs)).getlength(label))
        ty = y1 - fs - 2
        if ty < 2:
            ty = y1 + 2

        def _hits(px, py):
            return any(px - r <= p[0] <= px + tw + r and py - r <= p[1] <= py + fs * 1.15 + r
                       for p in (base, tip))

        tx = x1 + 2
        if _hits(tx, ty):                           # 左上角被圆点占了 → 右上角
            tx = max(x1 + 2, x2 - tw - 2)
            if _hits(tx, ty):                       # 还冲突 → 框内左下角
                tx, ty = x1 + 2, max(y1 + 2, y2 - fs - 4)
        texts.append((tx, ty, label, (35, 70, 255), fs, 0))

    return _pil_texts(vis, texts)


def _draw_overlay(vis, items, longest_text=None, marker_scale=1.0,
                  show_labels=True, style="simple"):
    """绘制结果图。style="simple"：蓝框 + 细黄线极简样式；
    style="detailed"：带中文标签与图例的详细样式。"""
    if style == "simple":
        return _draw_simple(vis, items, marker_scale)

    H, W = vis.shape[:2]
    s = max(H, W) / 1000.0                                  # 自适应系数（1000px 图≈1.0）
    lw = max(2, int(round(3.0 * s)))                        # 线宽
    r = max(5, int(round(14.0 * s * max(0.4, marker_scale))))   # 关键点半径
    fs = max(13, int(round(30.0 * s * max(0.7, marker_scale))))  # 字号
    texts = []

    for it in items:
        x1, y1, x2, y2 = it["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), lw, cv2.LINE_AA)

        path = it.get("path") or []
        if len(path) >= 2:
            pts = np.array([[int(p[0]), int(p[1])] for p in path], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 0, 0), lw + 5, cv2.LINE_AA)
            cv2.polylines(vis, [pts], False, (0, 255, 255), lw + 1, cv2.LINE_AA)
        else:
            base, tip = it["base"], it["tip"]
            _dashed_line(vis, base, tip, (0, 0, 0), lw + 5)
            _dashed_line(vis, base, tip, (0, 255, 255), lw + 1)

        base, tip = it["base"], it["tip"]
        for pt, color in ((base, (255, 80, 0)), (tip, (0, 220, 40))):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(vis, (cx, cy), r + max(2, lw), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), r, color, -1, cv2.LINE_AA)

        if show_labels:
            fs_kp = max(11, int(round(fs * 0.62)))
            for pt, name, rgb in ((base, "穗颈节", C_BASE_RGB), (tip, "穗尖", C_TIP_RGB)):
                tw = len(name) * fs_kp
                tx = int(pt[0]) + r + max(5, lw * 2)
                if tx + tw > W - 6:      # 右侧放不下就放左边
                    tx = max(6, int(pt[0]) - r - max(5, lw * 2) - tw)
                ty = int(pt[1]) - fs_kp // 2
                if ty < 6:               # 关键点贴顶，标签改到圆点下方
                    ty = int(pt[1]) + r + 4
                ty = max(0, min(H - fs_kp - 4, ty))
                texts.append((tx, ty, name, rgb, fs_kp, max(1, fs_kp // 16)))
            # 长度文字放在检测框内底部空白处（不与左下角图例打架）
            texts.append((x1 + max(4, lw), max(0, y2 - fs - 6),
                          it.get("len_text", ""), C_LEN_RGB, fs, max(1, fs // 16)))

    # 图例（左下角半透明面板）
    lfs = max(12, int(round(16.0 * s * min(1.4, max(0.7, marker_scale)))))
    row_h = int(lfs * 1.35)
    pad = int(lfs * 0.5)
    legend = [("穗颈节（蓝点）", C_BASE_RGB, (255, 80, 0)),
              ("穗尖（绿点）", C_TIP_RGB, (0, 220, 40)),
              ("主穗轴曲线穗长（黄线）", C_SPINE_RGB, (0, 255, 255))]
    if longest_text:
        legend.append((longest_text, C_LONGEST_RGB, (60, 60, 255)))
    box_h = pad * 2 + row_h * len(legend)
    box_w = pad * 3 + int(lfs * 1.1) + max(len(t) for t, _, _ in legend) * lfs
    ov = vis.copy()
    cv2.rectangle(ov, (0, max(0, H - box_h)), (min(W, box_w), H), (0, 0, 0), -1)
    vis = cv2.addWeighted(ov, 0.62, vis, 0.38, 0)
    for k, (label, rgb_text, bgr_mark) in enumerate(legend):
        cy = H - box_h + pad + row_h * k + row_h // 2
        cv2.circle(vis, (pad * 2, cy), max(4, lfs // 3), bgr_mark, -1, cv2.LINE_AA)
        texts.append((pad * 3 + int(lfs * 1.1), cy - lfs // 2,
                      label, rgb_text, lfs, max(1, lfs // 16)))

    return _pil_texts(vis, texts)


def keypoints_from_mask(mask):
    """模型关键点缺失时的几何兜底：
    穗尖 = 掩膜最高点；穗颈节 = 由茎秆（窄）向上进入穗体（宽）的宽度突变行，
    而不是一路取到裸茎最底部。返回 ((tip_x, tip_y), (neck_x, neck_y)) 或 (None, None)。"""
    if mask is None or mask.size == 0:
        return None, None
    widths = (mask > 0).sum(axis=1)
    rows = np.nonzero(widths)[0]
    if len(rows) < 4:
        return None, None

    top_y, bot_y = int(rows[0]), int(rows[-1])
    tip_x = float(np.nonzero(mask[top_y])[0].mean())

    # 茎秆宽度：取最下面 15% 行的中位宽度
    n_tail = max(3, int(len(rows) * 0.15))
    tail = widths[rows[-n_tail:]]
    stem_w = float(np.median(tail[tail > 0])) if (tail > 0).any() else 1.0
    thr = max(stem_w * 1.5, stem_w + 6)

    neck_y = bot_y
    for y in range(bot_y, top_y - 1, -1):
        if widths[y] > thr:
            neck_y = int(y)
            break
    neck_x = float(np.nonzero(mask[neck_y])[0].mean())
    return (tip_x, float(top_y)), (neck_x, float(neck_y))


def analyze_image(model, img_bgr, scale, conf, marker_scale=1.0, show_labels=True,
                  style="simple"):
    """对整张图做检测+穗长测量，返回 (vis_bgr, rows, longest_row)。

    marker_scale：关键点/文字的显示倍数（页面缩图看不清时可调大）。
    style：出图风格，"simple"=蓝框/细黄线/绿点/蓝点， "detailed"=加中文标签与图例。
    """
    res = model(img_bgr, conf=conf, verbose=False)[0]
    rows, items = [], []
    vis = img_bgr.copy()
    H, W = img_bgr.shape[:2]

    boxes = (res.boxes.xyxy.cpu().numpy()
             if getattr(res, "boxes", None) is not None else np.empty((0, 4)))
    kd = kconf = None
    if getattr(res, "keypoints", None) is not None and res.keypoints.data is not None:
        kd = res.keypoints.data.cpu().numpy()
        try:
            kconf = res.keypoints.conf.cpu().numpy()
        except Exception:
            kconf = None

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]

        # ---- 关键点：优先模型输出；缺失或置信度过低时用检测框几何兜底 ----
        kp_src, c_base, c_tip = "模型关键点", 0.0, 0.0
        if kd is not None and kd.shape[1] >= 2 and kd.shape[0] > i:
            p0 = (float(kd[i][0][0]), float(kd[i][0][1]))
            p1 = (float(kd[i][1][0]), float(kd[i][1][1]))
            if kconf is not None and kconf.shape[1] >= 2 and kconf.shape[0] > i:
                c0, c1 = float(kconf[i][0]), float(kconf[i][1])
                c_base, c_tip = max(c0, c1), min(c0, c1)
            # 方向校正：穗颈节在下（y 大），穗尖在上（y 小）
            if p0[1] < p1[1]:
                p_base, p_tip = p1, p0
            else:
                p_base, p_tip = p0, p1
            if min(c_base, c_tip) < 0.15:
                p_base = ((x1 + x2) / 2.0, float(y2))
                p_tip = ((x1 + x2) / 2.0, float(y1))
                kp_src = "检测框回退（关键点置信度低）"
        else:
            p_base = ((x1 + x2) / 2.0, float(y2))
            p_tip = ((x1 + x2) / 2.0, float(y1))
            kp_src = "检测框回退（无关键点输出）"

        pad_side = 30
        pad_bottom = 450
        cx1 = max(0, x1 - pad_side)
        cy1 = max(0, y1 - pad_side)
        cx2 = min(W, x2 + pad_side)
        cy2 = min(H, y2 + pad_bottom)
        crop = img_bgr[cy1:cy2, cx1:cx2]

        path_abs, curve_px = [], 0.0
        if crop.size > 0 and kp_src == "模型关键点":
            mask = segment_panicle(crop)
            skel = skeletonize_mask(mask)
            _, path = main_axis_from_skel(skel)
            b_crop, t_crop = refine_keypoints(
                mask, crop, path,
                (p_base[0] - cx1, p_base[1] - cy1),
                (p_tip[0] - cx1, p_tip[1] - cy1))
            p_base = (b_crop[0] + cx1, b_crop[1] + cy1)
            p_tip = (t_crop[0] + cx1, t_crop[1] + cy1)
            # 主穗轴优先用平滑弧线；弧线失效时回退骨架直径路径
            spine = spine_from_mask(mask, b_crop, t_crop)
            path_crop = spine if len(spine) >= 2 else trim_path_to_kpts(path, b_crop, t_crop)
            path_abs = [(float(px) + cx1, float(py) + cy1) for px, py in path_crop]
            curve_px = path_len(path_crop) if len(path_crop) >= 2 else euclidean_px(p_base, p_tip)
        elif crop.size > 0:
            # 关键点回退：用穗体掩膜把端点贴到穗尖与「茎秆→穗体」分界处
            mask = segment_panicle(crop)
            t_c, b_c = keypoints_from_mask(mask)
            if t_c is not None:
                p_tip = (t_c[0] + cx1, t_c[1] + cy1)
                p_base = (b_c[0] + cx1, b_c[1] + cy1)
            curve_px = euclidean_px(p_base, p_tip)
            path_abs = [(p_tip[0], p_tip[1]), (p_base[0], p_base[1])]
        else:
            curve_px = euclidean_px(p_base, p_tip)

        line_px = euclidean_px(p_base, p_tip)
        conf_kp = round(min(c_base, c_tip), 3) if kp_src == "模型关键点" else 0.0
        rows.append({
            "穗号": i + 1,
            "直线长度(px)": round(line_px, 2),
            "直线穗长(cm)": round(line_px * scale, 2),
            "曲线长度(px)": round(curve_px, 2),
            "曲线穗长(cm)": round(curve_px * scale, 2),
            "穗颈节(px)": f"({p_base[0]:.0f}, {p_base[1]:.0f})",
            "穗尖(px)": f"({p_tip[0]:.0f}, {p_tip[1]:.0f})",
            "关键点置信度": conf_kp,
            "关键点来源": kp_src,
            # 下划线开头：仅供界面做局部放大，不进表格 / CSV
            "_base": (float(p_base[0]), float(p_base[1])),
            "_tip": (float(p_tip[0]), float(p_tip[1])),
        })
        items.append({
            "box": (x1, y1, x2, y2),
            "base": p_base,
            "tip": p_tip,
            "path": path_abs,
            "len_text": f"{curve_px * scale:.1f}cm",
        })

    longest = max(rows, key=lambda r: r["曲线长度(px)"]) if rows else None
    longest_text = (f"最长穗 = {longest['曲线穗长(cm)']:.2f} cm" if longest else None)
    if items:
        vis = _draw_overlay(vis, items, longest_text, marker_scale, show_labels, style)
    return vis, rows, longest
