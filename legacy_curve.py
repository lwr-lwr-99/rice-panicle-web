# -*- coding: utf-8 -*-
"""
桥接模块：把原始测穗长脚本 panicle_longest/scripts/05_infer_curve.py 接到网页上。

- 直接用 importlib 按文件路径加载原始脚本，不复制、不改写其中的算法；
- 原始脚本被改动（保存新版本）后会自动重新加载，网页无需重启；
- 输出统一成网页表格需要的字段。

原始脚本位置可用环境变量 PANICLE_LEGACY_SCRIPT 覆盖。
"""
import importlib.util
import os

_DEFAULT_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "05_infer_curve.py",
)
SCRIPT_PATH = os.environ.get("PANICLE_LEGACY_SCRIPT", _DEFAULT_SCRIPT)

_state = {"mtime": None, "mod": None}


def script_info():
    """返回 (脚本路径, 是否存在)。"""
    return SCRIPT_PATH, os.path.exists(SCRIPT_PATH)


def load_script():
    """按文件路径加载原始脚本；文件改动后自动重载。"""
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError(f"找不到原始测穗长脚本：{SCRIPT_PATH}")
    mtime = os.path.getmtime(SCRIPT_PATH)
    if _state["mod"] is None or _state["mtime"] != mtime:
        spec = importlib.util.spec_from_file_location("panicle_curve_script", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _state["mod"] = mod
        _state["mtime"] = mtime
    return _state["mod"]


def analyze_legacy(model, img_bgr, scale, conf, marker_scale=1.0):
    """调用原始脚本的 measure_image，返回与内置管线同构的 (vis, rows, longest_row)。"""
    mod = load_script()
    if not hasattr(mod, "measure_image"):
        raise AttributeError(
            "原始脚本里没有 measure_image(img, model, scale, conf) 函数，"
            "请先更新 05_infer_curve.py（已提供可直接调用的版本）。")

    vis, raw_rows, longest = mod.measure_image(
        img_bgr, model, scale, conf, marker_scale=marker_scale, draw=True)

    rows = []
    for r in raw_rows:
        base, tip = r.get("base"), r.get("tip")
        rows.append({
            "穗号": r.get("panicle_id"),
            "直线长度(px)": "",
            "直线穗长(cm)": r.get("line_cm"),
            "曲线长度(px)": "",
            "曲线穗长(cm)": r.get("curve_cm"),
            "穗颈节(px)": f"({base[0]:.0f}, {base[1]:.0f})" if base else "",
            "穗尖(px)": f"({tip[0]:.0f}, {tip[1]:.0f})" if tip else "",
            "关键点置信度": "",
            "关键点来源": "原始脚本 05_infer_curve.py",
            # 下划线开头：仅供界面局部放大，不进表格 / CSV
            "_base": base,
            "_tip": tip,
        })

    longest_row = None
    if longest:
        longest_row = {"穗号": longest.get("panicle_id"),
                       "曲线穗长(cm)": longest.get("curve_cm")}
    return vis, rows, longest_row
