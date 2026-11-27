# -*- coding: utf-8 -*-
"""
05_infer_curve.py - 沿穗弯曲的真实曲线长度
流程：pose检测位置 -> 灰度分割 -> 骨架化 -> 沿主穗轴算曲线长度
"""
import argparse, os, csv, cv2, numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize


def segment_panicle(crop):
    """黑背景下灰度分割稻穗"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 取最大连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == big, 255, 0).astype(np.uint8)
    return mask


def skeletonize_mask(mask):
    """用 skimage 细化骨架"""
    skel = skeletonize(mask > 0)
    return (skel * 255).astype(np.uint8)


def main_axis_from_skel(skel):
    """
    找骨架上最长的路径（树的直径）= 主穗轴
    方法：两次BFS
    1. 从任意点出发，找最远点A
    2. 从A出发，找最远点B，A到B的路径就是直径（最长路径）
    """
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 2:
        return 0.0, []

    from collections import deque

    def bfs(start):
        """BFS找最远点和路径"""
        dist = {start: 0.0}
        prev = {start: None}
        q = deque([start])
        far = start
        far_d = 0.0
        while q:
            cx, cy = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx+dx, cy+dy)
                    if nb not in pts or nb in dist:
                        continue
                    w = 1.41421356 if dx != 0 and dy != 0 else 1.0
                    dist[nb] = dist[(cx, cy)] + w
                    prev[nb] = (cx, cy)
                    q.append(nb)
                    if dist[nb] > far_d:
                        far_d = dist[nb]
                        far = nb
        return far, far_d, prev

    # 第一次BFS：从任意点找最远点A
    start_point = next(iter(pts))
    A, _, _ = bfs(start_point)

    # 第二次BFS：从A出发找最远点B，A到B就是最长路径
    B, max_len, prev = bfs(A)

    # 回溯路径
    path = []
    cur = B
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return max_len, path


def trim_path_to_kpts(path, p_base, p_tip):
    """把主穗轴路径裁剪到穗基~穗尖之间。

    稻穗结构上，穗长只应量到穗基（稻穗下方凸起的节/穗颈节）为止，
    节以下的茎秆不属于穗。取路径上离穗基、穗尖最近的两个点，
    保留两点之间的骨架段作为穗轴曲线。"""
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
    """按 8 邻接加权计算路径长度（水平/垂直=1，对角=sqrt(2)）"""
    total = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        total += 1.41421356 if dx != 0 and dy != 0 else 1.0
    return total


def smooth_path(path, p0=None, p1=None):
    """2次多项式拟合主穗轴，平滑弧线，连接两个点，无折线。"""
    if len(path) < 10:
        return path
    arr = np.array(path, dtype=np.float64)
    order = np.argsort(arr[:, 1])
    arr = arr[order]
    ys = arr[:, 1]
    xs = arr[:, 0]
    # 按 y 分组取中位数，去掉侧枝干扰
    y_unique = np.unique(ys)
    x_med = np.array([np.median(xs[ys == yy]) for yy in y_unique])
    # 把蓝点绿点加入拟合点，确保端点无折线
    if p0 is not None:
        y_unique = np.append(y_unique, p0[1])
        x_med = np.append(x_med, p0[0])
    if p1 is not None:
        y_unique = np.append(y_unique, p1[1])
        x_med = np.append(x_med, p1[0])
    # 重新排序
    order2 = np.argsort(y_unique)
    y_unique = y_unique[order2]
    x_med = x_med[order2]
    # 加权拟合，端点权重大，既接上又平滑无折线
    weights = np.ones_like(y_unique)
    if p0 is not None:
        weights[-1] = 100  # 蓝点权重100倍
    if p1 is not None:
        weights[0] = 100   # 绿点权重100倍
    coeffs = np.polyfit(y_unique, x_med, 3, w=weights)
    poly = np.poly1d(coeffs)
    # 从蓝点到绿点
    if p0 is not None and p1 is not None:
        y_max = p0[1]
        y_min = p1[1]
    else:
        y_min, y_max = y_unique.min(), y_unique.max()
    y_new = np.linspace(y_min, y_max, 100)
    x_new = poly(y_new)
    pts = [(float(xi), float(yi)) for xi, yi in zip(x_new, y_new)]
    return pts



def smooth_polyline(path, p0=None, p1=None, n=200, win=15):
    """沿骨架路径重采样 + 滑动平均平滑，两端用 smoothstep 精确落到穗颈节/穗尖。

    与 smooth_path 的 x=f(y) 三次拟合不同：这里不按行去拟合，而是顺着
    穗轴的实际走向（含强弯穗）等弧长重采样后再平滑，所以
      1) 弧长不会被"压扁"，曲线长度始终 >= 两点直线距离；
      2) 线条圆滑不抖（滑动平均去掉骨架锯齿）；
      3) 首尾严格落在蓝点/绿点上，黄线不会从点上拐出去。
    """
    if not path or len(path) < 10:
        return path
    arr = np.array(path, dtype=np.float64)
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return path
    si = np.linspace(0, s[-1], n)
    xs = np.interp(si, s, arr[:, 0])
    ys = np.interp(si, s, arr[:, 1])
    # 滑动平均（边界用对称延拓，避免两端被拉偏）
    k = max(3, int(win) // 2 * 2 + 1)
    pad = k // 2
    xs_p = np.pad(xs, pad, mode="edge")
    ys_p = np.pad(ys, pad, mode="edge")
    xs = np.convolve(xs_p, np.ones(k) / k, mode="valid")
    ys = np.convolve(ys_p, np.ones(k) / k, mode="valid")
    # 端点用 smoothstep 渐入，保证严格经过 p0 / p1
    if p0 is not None:
        x0, y0 = float(p0[0]), float(p0[1])
        m = int(n * 0.14)
        idx = np.arange(m)
        t = idx / max(1, m - 1)
        w = t * t * (3 - 2 * t)
        xs[:m] = (1 - w) * x0 + w * xs[:m]
        ys[:m] = (1 - w) * y0 + w * ys[:m]
        xs[0], ys[0] = x0, y0
    if p1 is not None:
        x1, y1 = float(p1[0]), float(p1[1])
        m = int(n * 0.14)
        idx = np.arange(m)
        t = idx / max(1, m - 1)
        w = t * t * (3 - 2 * t)
        xs[n - m:] = w[::-1] * x1 + (1 - w[::-1]) * xs[n - m:]
        ys[n - m:] = w[::-1] * y1 + (1 - w[::-1]) * ys[n - m:]
        xs[-1], ys[-1] = x1, y1
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _bright_x_in_row(gray_crop, mask, y, fallback_x):
    """该行最大连续前景段内灰度最大的像素 x（茎秆/稻穗最亮处），
    避免闭运算填充的暗缝导致中点落背景。"""
    xs = np.nonzero(mask[y])[0]
    if len(xs) == 0:
        return fallback_x
    segs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    longest = max(segs, key=len)
    vals = gray_crop[y, longest]
    return float(longest[int(np.argmax(vals))])


def find_grain_top(mask, path, tip_raw):
    """沿主穗轴路径从空须顶端往下找米粒开始的拐点。
    空须处主穗轴很细(中心几像素)，米粒处左右两侧都有米粒(局部宽度突增)。
    用主穗轴点左右各 15px 内的前景像素数判断，避免斜穗时整行 row_w 误判。"""
    h, w = mask.shape[:2]
    top_y = int(tip_raw[1])
    # 从 path 顶端往下，按 y 排序
    pts = sorted([p for p in path if p[1] >= top_y], key=lambda p: p[1])
    if not pts:
        return tip_raw
    # 先看顶端局部宽度
    def local_w(px, py):
        x0 = max(0, int(px)-15); x1 = min(w, int(px)+16)
        return int((mask[py, x0:x1] > 0).sum())
    # 顶端局部已经 >12 说明没空须
    if local_w(tip_raw[0], top_y) > 8:
        return tip_raw
    for px, py in pts:
        py = int(py)
        if py >= h: continue
        lw = local_w(px, py)
        if lw > 8:
            return (float(px), float(py))
    return tip_raw


def find_stem_node(mask, crop_img, my, max_search=900):
    """在稻穗着粒区下缘(my)下方的**裸茎段**中找最明显的突起节（穗颈节）。

    旧版把着粒区的宽行也算进了中位数统计，med_w 被严重抬高，判定阈值
    max(med_w*1.1, med_w+2) 基本达不到 → 节全程识别失败，穗颈节只好退回
    谷粒下缘，蓝点就停在了着粒区下缘、没到穗颈节。现在改为：
      1) 先定位裸茎段起点 stem_start 与末端 stem_end；
      2) 中位数只在裸茎段内统计，节 = 宽度局部极大且 >= med_w*1.08；
      3) 只在裸茎段上半段找（穗颈节紧邻稻穗，不会跑到茎秆末端）；
      4) 找不到明显突起时退回裸茎段 30% 处，而不是退回谷粒下缘。"""
    H = mask.shape[0]
    # 每行：最大连续前景段长（茎秆真实宽度）与该段亮度
    seg_w = np.zeros(H, dtype=np.float64)
    seg_v = np.zeros(H, dtype=np.float64)
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    for y in range(H):
        xs = np.nonzero(mask[y])[0]
        if len(xs) == 0:
            continue
        segs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        longest = max(segs, key=len)
        seg_w[y] = len(longest)
        seg_v[y] = float(hsv[y, longest, 2].mean())
    max_w = seg_w.max()
    if max_w < 10:
        return None

    # ---- 1) 裸茎段起点：从 my+20 往下，连续10行宽度都够细 ----
    # 关键：着粒区最宽可达 150+px，旧阈值 0.3*max_w 会把稻穗下缘过渡区
    # 误判为茎秆起点；细茎秆一般为着粒区的 15%-25%，故优先用 0.2。
    stem_start = None
    for frac in (0.15, 0.2, 0.3):
        w_thr = max(max_w * frac, 8.0)
        y = my + 20
        while y < H - 10:
            if seg_w[y:y + 10].max() <= w_thr:
                stem_start = y
                break
            y += 1
        if stem_start is not None:
            break
    if stem_start is None:
        return None

    # ---- 2) 裸茎段末端：起点往下最后一个有前景的行 ----
    stem_end = stem_start
    y = stem_start
    while y < min(H, stem_start + max_search):
        if seg_w[y] > 0:
            stem_end = y
        y += 1
    stem_len = stem_end - stem_start
    if stem_len < 40:                      # 茎秆太短，没有可分辨的节
        return stem_start

    # ---- 3) 裸茎段内统计中位数，再找宽度局部极大 ----
    seg = seg_w[stem_start:stem_end + 1]
    if (seg > 0).sum() < 20:
        return stem_start
    med_w = float(np.median(seg[seg > 0]))
    if med_w < 6:
        return stem_start

    # ---- 3) 裸茎段内统计中位数，再找"最深的一个明显鼓包" ----
    seg = seg_w[stem_start:stem_end + 1]
    if (seg > 0).sum() < 20:
        return int(stem_start + 0.35 * stem_len)
    med_w = float(np.median(seg[seg > 0]))
    if med_w < 6:
        return int(stem_start + 0.35 * stem_len)

    k = np.ones(9) / 9.0
    seg_s = np.convolve(seg, k, mode="same")
    upper_end = stem_start + int(stem_len * 0.62)
    w_floor = med_w * 1.15                     # 明显宽于常规茎秆才算鼓包
    groups = []                                # [(y0, y1)]
    y = stem_start + max(8, int(0.12 * stem_len))
    while y < upper_end:
        if seg_s[y - stem_start] >= w_floor:
            y0 = y
            while y < upper_end and seg_s[y - stem_start] >= w_floor:
                y += 1
            groups.append((y0, y - 1))
        else:
            y += 1
    if groups:
        # 取最深（最靠下、离稻穗最远）的一组 —— 剑叶鞘叶片横压造成的高位
        # 假峰在上方，真正的穗颈节在更下面的鞘口鼓起处
        y0, y1 = groups[-1]
        loc = y0 + int(np.argmax(seg_s[y0 - stem_start:y1 - stem_start + 1]))
        return loc
    # ---- 4) 没有明显鼓包：退回裸茎段 35% 处（仍在茎秆上） ----
    return int(stem_start + 0.35 * stem_len)
    med_w = float(np.median(seg[seg > 0]))
    if med_w < 6:
        return int(stem_start + 0.35 * stem_len)

    # 平滑（去掉单行噪声与锯齿，避免抖动），在裸茎段上 60% 内找宽度峰
    k = np.ones(9) / 9.0
    seg_s = np.convolve(seg, k, mode="same")
    upper_end = stem_start + int(stem_len * 0.6)
    win = seg_s[6:upper_end - stem_start] if upper_end - stem_start > 10 else seg_s[6:]
    if win.size == 0:
        return int(stem_start + 0.35 * stem_len)
    off = 6
    peak_local = int(np.argmax(win)) + off
    peak_w = float(seg_s[peak_local])
    if peak_w >= med_w * 1.05 and seg_w[stem_start + peak_local] >= med_w * 1.02:
        return stem_start + peak_local
    # ---- 4) 没有明显鼓包：退回裸茎段 35% 处（仍在茎秆上） ----
    return int(stem_start + 0.35 * stem_len)


def refine_keypoints(mask, crop_img, path, pred_base, pred_tip):
    """把穗基/穗尖校正到稻穗主体上（crop 坐标系）：
    - 穗尖 = 主穗轴最上端（稻穗顶端）
    - 穗基 = 稻穗着粒区下缘，并继续下移到茎秆上最明显的突起节（梗/竹节），
      起点从下方"梗"（茎秆节）开始
    - x 取该行最大前景段内最亮像素，保证点落在稻穗本体上
    """
    if len(path) < 2:
        return pred_base, pred_tip

    tip_raw = min(path, key=lambda p: p[1])
    tip_y = find_grain_top(mask, path, tip_raw)
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    row_w = (mask > 0).sum(axis=1)
    max_w = int(row_w.max()) if row_w.size else 0
    if max_w < 5:
        return pred_base, (float(tip_y[0]), float(tip_y[1]))
    thr = max(10, int(max_w * 0.25))
    y_base = None
    for y in range(mask.shape[0] - 1, -1, -1):
        if row_w[y] >= thr:
            y_base = y
            break
    if y_base is None:
        y_base = int(pred_base[1])
    # 向下找茎秆最明显的突起节；找不到则用茎秆段起点
    node_y = find_stem_node(mask, crop_img, y_base)
    if node_y is not None:
        y_base = node_y
    # 蓝点横坐标：优先取主穗轴（骨架）在该行的 x，保证点落在茎秆中线上；
    # 骨架在该行缺失时再退回"该行最亮前景像素"
    near = [p for p in path if abs(p[1] - y_base) <= 12]
    if near:
        x_base = float(np.mean([p[0] for p in near]))
    else:
        x_base = _bright_x_in_row(gray_crop, mask, y_base, float(pred_base[0]))
    # 穗尖 x 用主穗轴在 tip_y 那一行的骨架 x（避免落到米粒上导致黄线拐弯）
    ty = float(tip_y[1])
    tip_path_pt = min(path, key=lambda p: abs(p[1] - ty))
    tip = (float(tip_path_pt[0]), ty)
    return (x_base, float(y_base)), tip


def measure_image(img_bgr, model, scale=0.009141, conf=0.5,
                  marker_scale=1.0, draw=True):
    """检测 + 直线/曲线穗长测量（供网页等外部程序直接调用）。

    参数：
        img_bgr      : BGR 原图（numpy 数组，不是路径）
        model        : 已加载的 YOLO 模型对象
        scale        : 标定系数 cm/px
        conf         : 检测置信度阈值
        marker_scale : 线宽/圆点/字号的放大倍数（4K 图在网页里缩得很小，可调大）
        draw         : 是否在图上画框/黄线/关键点
    返回：
        (vis, rows, longest)
        rows    : [{"panicle_id","line_cm","curve_cm","base","tip"}]
        longest : 曲线最长的那个 row（无检测结果时返回 None）
    """
    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy() if draw else None
    # 尺寸随分辨率自适应：固定 3px/10px 在 4K 图上缩到网页里几乎看不见
    s = max(H, W) / 1000.0
    lw = max(2, int(round(3.0 * s * max(0.6, marker_scale))))
    r = max(4, int(round(10.0 * s * max(0.4, marker_scale))))
    fs = max(0.6, min(1.8, 0.5 * s * max(0.7, marker_scale)))

    res = model(img_bgr, conf=conf)[0]
    rows = []

    for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 25
        pad_bottom = 450
        cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
        cx2 = min(W, x2+pad); cy2 = min(H, y2+pad_bottom)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        mask = segment_panicle(crop)
        skel = skeletonize_mask(mask)
        _, path = main_axis_from_skel(skel)

        # pose 端点坐标（原图坐标系），先按结构校正方向
        kp = res.keypoints.data.cpu().numpy()[i]
        k0 = (float(kp[0][0]), float(kp[0][1]))
        k1 = (float(kp[1][0]), float(kp[1][1]))
        if k0[1] < k1[1]:
            base_abs, tip_abs = k1, k0
        else:
            base_abs, tip_abs = k0, k1
        # 关键点吸附到稻穗主体（穗基=下方茎秆节/梗，穗尖=穗顶端）
        b_crop, t_crop = refine_keypoints(
            mask, crop, path,
            (float(base_abs[0]) - cx1, float(base_abs[1]) - cy1),
            (float(tip_abs[0]) - cx1, float(tip_abs[1]) - cy1))
        p_base = (b_crop[0] + cx1, b_crop[1] + cy1)
        p_tip = (t_crop[0] + cx1, t_crop[1] + cy1)

        line_px = float(np.hypot(p_base[0]-p_tip[0], p_base[1]-p_tip[1]))
        path = trim_path_to_kpts(path, b_crop, t_crop)
        draw_path = smooth_polyline(path, b_crop, t_crop)
        # 曲线长度按"实际画出来的那条平滑曲线"来算：它从蓝点出发、经穗轴、
        # 到绿点结束，必然 >= 两点直线距离，不会出现"曲线比直线还短"的矛盾
        if len(draw_path) >= 2:
            curve_px = 0.0
            for j in range(len(draw_path) - 1):
                curve_px += float(np.hypot(draw_path[j+1][0]-draw_path[j][0],
                                           draw_path[j+1][1]-draw_path[j][1]))
        else:
            path = trim_path_to_kpts(path, b_crop, t_crop)
            curve_px = path_len(path) if len(path) >= 2 else line_px

        line_cm = line_px * scale
        curve_cm = curve_px * scale

        rows.append({"panicle_id": i+1, "line_cm": round(line_cm, 2),
                     "curve_cm": round(curve_cm, 2),
                     "base": (float(p_base[0]), float(p_base[1])),
                     "tip": (float(p_tip[0]), float(p_tip[1])),
                     "box": (cx1, cy1, cx2, cy2)})

        if not draw:
            continue

        # 可视化：框 + 黄色弧线(穗基~穗尖，平滑) + 蓝点穗基 + 绿点穗尖
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 0), lw)
        draw_path = smooth_path(path, b_crop, t_crop)
        if len(draw_path) >= 2:
            pts = np.array([[int(px) + cx1, int(py) + cy1] for px, py in draw_path], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 255), lw)
        else:
            cv2.line(vis, (int(p_base[0]), int(p_base[1])),
                     (int(p_tip[0]), int(p_tip[1])), (0, 255, 255), lw)
        cv2.circle(vis, (int(p_base[0]), int(p_base[1])), r, (255, 0, 0), -1)   # 蓝点 = 穗基(节)
        cv2.circle(vis, (int(p_tip[0]), int(p_tip[1])), r, (0, 255, 0), -1)      # 绿点 = 穗尖
        cv2.putText(vis, f"curve:{curve_cm:.1f}cm", (cx1, max(int(fs * 32), cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), lw)

    if not rows:
        return vis, rows, None

    longest = max(rows, key=lambda x: x["curve_cm"])
    if draw:
        lfs = max(0.8, min(2.0, 1.2 * s * max(0.7, marker_scale) * 0.6))
        cv2.putText(vis, f"LONGEST curve = {longest['curve_cm']:.2f} cm",
                    (20, max(int(lfs * 34), 40)), cv2.FONT_HERSHEY_SIMPLEX,
                    lfs, (255, 0, 0), lw)
    return vis, rows, longest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--scale", type=float, default=0.009141)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = YOLO(args.weights)

    arr = np.fromfile(args.image, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # CLI 保持原来的固定尺寸观感：marker_scale=0.33 → 1000px 图上线宽≈3、点半径≈10
    vis, rows, longest = measure_image(img, model, args.scale, args.conf,
                                       marker_scale=0.33, draw=True)
    if not rows:
        print("未检测到稻穗"); return

    print(f"共 {len(rows)} 个稻穗")
    for r in rows:
        print(f"  穗{r['panicle_id']}: 直线 {r['line_cm']:.2f}cm, 曲线 {r['curve_cm']:.2f}cm")
    print(f"最长穗：第 {longest['panicle_id']} 个，曲线长度 {longest['curve_cm']:.2f} cm")

    out_img = os.path.join(args.outdir, "curve_" + os.path.basename(args.image))
    ok, buf = cv2.imencode(".jpg", vis)
    if ok:
        buf.tofile(out_img); print("结果图：", out_img)

    out_csv = os.path.join(args.outdir, "curve_" +
                           os.path.splitext(os.path.basename(args.image))[0] + ".csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["panicle_id", "line_cm", "curve_cm"])
        w.writeheader()
        w.writerows([{k: v for k, v in r.items() if k in ("panicle_id", "line_cm", "curve_cm")}
                     for r in rows])
    print("穗长明细：", out_csv)


if __name__ == "__main__":
    main()
