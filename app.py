# -*- coding: utf-8 -*-
"""
水稻穗长智能测量与最长穗筛选系统（比赛演示版）
------------------------------------------------
面向农科院育种研究与大田推广：YOLOv11n-Pose 关键点检测 + 主穗轴平滑弧线提取。

运行：
    conda activate pytorch_work
    streamlit run app.py
"""
import os
import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

def _pipeline_module():
    """加载 pipeline.py，并在检测到文件被改动时自动热重载。

    原因：Streamlit 只重跑主脚本，已 import 的本地模块会停在内存里的旧版本，
    改了 pipeline.py 而不重启，就会出现「analyze_image() 收到意外关键字参数」这类报错。
    """
    import importlib
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    mtime = os.path.getmtime(os.path.join(here, "pipeline.py"))
    mod = sys.modules.get("pipeline")
    if mod is None:
        import pipeline as mod  # noqa: F401
    elif getattr(_pipeline_module, "_mtime", None) != mtime:
        mod = importlib.reload(mod)
    _pipeline_module._mtime = mtime
    return mod


_pl = _pipeline_module()
analyze_image, DEFAULT_SCALE = _pl.analyze_image, _pl.DEFAULT_SCALE


def _legacy_module():
    """桥接模块 legacy_curve.py（同样跟随文件改动自动重载）。"""
    import importlib
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    mtime = os.path.getmtime(os.path.join(here, "legacy_curve.py"))
    mod = sys.modules.get("legacy_curve")
    if mod is None:
        import legacy_curve as mod  # noqa: F401
    elif getattr(_legacy_module, "_mtime", None) != mtime:
        mod = importlib.reload(mod)
    _legacy_module._mtime = mtime
    return mod


_legacy = _legacy_module()
analyze_legacy = _legacy.analyze_legacy
LEGACY_SCRIPT, LEGACY_EXISTS = _legacy.script_info()

# ----------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DEFAULT = os.path.join(APP_DIR, "best.pt")
DEMO_IMAGE = os.path.join(APP_DIR, "sample.jpg")
USE_LEGACY = True          # True：直接调用 panicle_longest/scripts/05_infer_curve.py 的算法

# ----------------------------------------------------------------------
# 页面配置与视觉样式
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="水稻穗长智能测量与最长穗筛选系统",
    page_icon="🌾",
    layout="wide",
)

st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #66bb6a 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: #fff;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 10px 0; font-size: 30px; font-weight: 800; }
    .hero p { margin: 0; font-size: 15px; opacity: .95; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px;
        padding: 4px 12px;
        margin: 10px 8px 0 0;
        font-size: 13px;
    }
    .section-title { color: #1b5e20; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# 顶部 Hero
st.markdown("""
<div class="hero">
    <h1>🌾 水稻穗长智能测量与最长穗筛选系统</h1>
    <p>基于 YOLOv11n-Pose 关键点检测 + 主穗轴平滑弧线提取 · 服务农科院育种研究，兼顾大田与大众推广</p>
    <span class="badge">🧑‍🌾 智能农业装备</span>
    <span class="badge">🧠 图像深度学习</span>
    <span class="badge">🎨 交互与可视化</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 侧边栏：系统设置
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 系统设置")

    st.subheader("模型配置")
    model_path = st.text_input(
        "模型权重路径",
        value=WEIGHTS_DEFAULT,
        help="训练好的 YOLOv11n-pose 模型权重（best.pt），默认与应用同目录",
    )
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(APP_DIR, model_path)

    st.subheader("尺寸标定")
    scale = st.number_input(
        "标定系数（cm/px）",
        min_value=0.0001,
        max_value=1.0,
        value=DEFAULT_SCALE,
        step=0.0001,
        format="%.6f",
        help="实际厘米 ÷ 对应像素数。默认 0.009141（5cm≈547px）；机位、焦距或参照物变化时，"
             "请用直尺/标尺重新标定，否则厘米尺寸整体失真。",
    )

    conf_threshold = st.slider(
        "检测置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    marker_scale = st.slider(
        "关键点与线条大小倍数",
        min_value=0.6,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="原图分辨率很高（如 4K）时，网页缩略图会把标记缩得很小，调大即可看清。",
    )

    st.divider()
    st.caption("💡 穗长口径：黄色曲线为沿主穗轴的曲线穗长；直线穗长（穗基—穗尖两点距离）作为复核基线。")
    st.caption("⚠️ 曲线穗长算法仍在迭代优化，正式比对以人工实测口径为准。")


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None


model = load_model(model_path)

# ----------------------------------------------------------------------
# 主界面：四个标签页
# ----------------------------------------------------------------------
tab_demo, tab_method, tab_metric, tab_guide = st.tabs(
    ["🔍 在线演示", "🧠 技术方案", "📊 模型指标", "📖 使用说明"]
)

# -------------------- 标签页 1：在线演示 --------------------
with tab_demo:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📷 上传稻穗图片")
        use_demo = st.checkbox("使用内置示例图（FCMU382_3）", value=False)
        uploaded_file = st.file_uploader(
            "选择图片文件",
            type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            help="建议平整背景、均匀光照；稻穗与茎秆完整入镜，穗尖不要被裁切。",
        )

        img_bgr = None
        if use_demo and os.path.exists(DEMO_IMAGE):
            arr = np.fromfile(DEMO_IMAGE, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     caption="原始图片", use_container_width=True)
        else:
            st.info("👆 请上传图片，或勾选内置示例图快速体验。")

    with col_right:
        st.subheader("🔍 分析结果")
        if model is None:
            st.error(f"❌ 未找到模型文件：`{model_path}`")
            st.info("请将训练好的 `best.pt` 放在应用同目录，或在侧边栏填写正确路径。")
        elif img_bgr is None:
            st.info("👆 请先在左侧提供图片。")
        else:
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                with st.spinner("正在检测稻穗关键点并追踪主穗轴……"):
                    try:
                        if LEGACY_EXISTS:
                            # 直接调用你的测穗长脚本（panicle_longest/scripts/05_infer_curve.py）
                            vis, rows, longest = analyze_legacy(
                                model, img_bgr, scale, conf_threshold,
                                marker_scale=marker_scale)
                        else:
                            raise FileNotFoundError(
                                f"找不到测穗长脚本：{LEGACY_SCRIPT}")
                    except Exception as e:
                        st.error(f"推理失败：{e}")
                        rows, vis, longest = [], None, None

                if vis is not None and rows:
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption=f"识别与穗长测量结果（共 {len(rows)} 穗）",
                             use_container_width=True)

                    avg_line = float(np.mean([r["直线穗长(cm)"] for r in rows]))
                    avg_curve = float(np.mean([r["曲线穗长(cm)"] for r in rows]))
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("检测穗数", len(rows))
                    k2.metric("平均直线穗长", f"{avg_line:.2f} cm")
                    k3.metric("平均曲线穗长", f"{avg_curve:.2f} cm")
                    k4.metric("最长穗长", f"{longest['曲线穗长(cm)']:.2f} cm")

                    rows_show = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                    df = pd.DataFrame(rows_show)
                    df["标记"] = ["★ 最长穗" if r["穗号"] == longest["穗号"] else "" for r in rows_show]

                    with st.expander("🔎 关键点局部放大（核对穗颈节 / 穗尖）", expanded=False):
                        st.caption("逐穗截取穗颈节与穗尖周围区域，蓝点为穗颈节、绿点为穗尖，便于确认关键点是否落在正确位置。")
                        zoom_cols = st.columns(len(rows))
                        h_img, w_img = vis.shape[:2]
                        half = int(max(60, min(h_img, w_img) * 0.055))
                        for col, r in zip(zoom_cols, rows):
                            with col:
                                for name, key in (("穗颈节（蓝点）", "_base"), ("穗尖（绿点）", "_tip")):
                                    pt = r.get(key)
                                    if not pt:
                                        continue
                                    px, py = pt
                                    y0 = max(0, int(py) - half)
                                    x0 = max(0, int(px) - half)
                                    crop = vis[y0:int(py) + half, x0:int(px) + half]
                                    if crop.size:
                                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                                 caption=f"穗{r['穗号']} · {name}",
                                                 use_container_width=True)

                    with st.expander("📋 查看逐穗明细表"):
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    st.subheader("💾 导出结果")
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                    st.download_button(
                        "📥 下载穗长明细 CSV",
                        data=csv_buf.getvalue().encode("utf-8-sig"),
                        file_name="穗长测量明细.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    ok, buf = cv2.imencode(".jpg", vis)
                    if ok:
                        st.download_button(
                            "📥 下载结果图",
                            data=buf.tobytes(),
                            file_name="穗长识别结果.jpg",
                            mime="image/jpeg",
                            use_container_width=True,
                        )
                elif vis is None:
                    pass
                else:
                    st.warning("⚠️ 未检测到稻穗，可尝试降低置信度阈值，或检查图片与模型权重。")

# -------------------- 标签页 2：技术方案（三专家视角） --------------------
with tab_method:
    st.markdown('<p class="section-title">🧑‍🌾 从智能农业装备视角：育种场景与落地路径</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**育种科研需求**
- 穗长是水稻株型、产量潜力与抗倒性评价的核心考种性状之一。
- 传统人工用直尺/卡尺逐穗测量，记录慢、主观性强，大群体考种效率低。
- 系统面向农科院育种材料初选、穗长表型批量采集与种质资源建档。

**测量口径对接人工实测**
- 起点：穗颈节（茎秆上突起的节），不把下方裸茎秆计入穗长。
- 终点：最顶端穗尖（含米粒开始的拐点校正）。
- 支持标尺标定，测量值可与人工直尺实测逐项复核比对。
        """)
    with c2:
        st.markdown("""
**工程化落地考虑**
- 单株/单穗图像上传即可出结果，无需专业操作门槛，便于基层与大众使用。
- 结果图 + 明细表 + CSV 导出，直接对接育种台账与论文数据整理。
- 后续可扩展：田间多穗批量网格识别、穗长分布统计、品种穗长档案库。

**推广定位**
- 科研端：辅助育种家提高考种效率，沉淀标准化表型数据。
- 大众端：轻量易用的穗长测量演示，降低智慧农业技术的认知门槛。
        """)
    with c3:
        st.markdown("""
**当前边界（诚实说明）**
- 曲线穗长基于主穗轴平滑弧线提取，算法仍在优化。
- 与农科院人工实测的系统误差（MAE/RMSE）尚待补充比对。
- 拍摄机位、背景杂乱、穗子相互遮挡时，测量结果需人工复核。
        """)

    st.divider()
    st.markdown('<p class="section-title">🧠 从图像深度学习视角：模型与测量管线</p>',
                unsafe_allow_html=True)
    st.markdown("""
**1. 数据与标注**
- 采集真实稻穗图像 450 张，剔除空标、重复与损坏样本后有效样本 438 张；按 263 / 88 / 87 划分训练集、验证集、测试集。
- 单类别 `panicle`，每穗标注 2 个关键点：穗颈节（穗基）、穗尖。

**2. 模型选型：YOLOv11n-Pose**
- 轻量化姿态估计网络，同时输出检测框与关键点坐标，适合 CPU 端本地推理演示。
- 训练配置：epochs=50，imgsz=640，batch=8，SGD，lr0=0.01，mosaic=1.0，close_mosaic=10。

**3. 后处理测量管线**
1. 按检测框向下扩展裁剪框，纳入茎秆以定位穗颈节；
2. 灰度阈值分割 + 形态学闭运算，取最大连通域得到稻穗主体掩膜；
3. 在穗体骨架上取穗基—穗尖最短连通路径，沿中央主穗轴上行、不绕侧枝梗；
4. 逐行取该穗穗体（同一行属于该穗的谷粒整体）中点作为穗轴中心，再用稳健低阶多项式（4 次）拟合、按 MAD 迭代剔除芒须/邻穗造成的跳点，输出圆滑且贴合穗轴弯曲的主穗轴弧线（首尾以 smoothstep 平滑过渡到穗尖 / 穗颈节）；
5. 关键点校正：穗尖吸附到米粒拐点，穗基保留模型定位并横向对齐茎秆/穗体中心；
6. 沿平滑弧线按真实欧氏距离计算曲线像素长度（骨架直径路径仅在弧线失效时回退）；
7. 乘以标定系数（cm/px）换算为厘米；另给两点直线距离作为复核基线。
""")

    st.divider()
    st.markdown('<p class="section-title">🎨 从交互与可视化视角：界面与结果表达</p>',
                unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("""
**结果可视化约定（图例统一）**
- 🟦 检测框：识别到的稻穗范围（简洁样式蓝框，详细样式绿框）。
- 🟡 黄色曲线：沿主穗轴的穗长测量路径。
- 🔵 蓝色点：穗基（穗颈节）。
- 🟢 绿色点：穗尖。
- 🔤 左上角蓝色数字：该穗曲线穗长（cm）。
        """)
    with cc2:
        st.markdown("""
**界面层级**
- 顶部 Hero 概括项目定位与三个视角标签；侧边栏集中放置模型、标定与阈值设置。
- 演示区双栏对照原图与结果图；KPI 卡片先给汇总，明细表供逐穗核查。
- 配色以农业绿为主、米金为辅，指标与导出操作就近放置，符合比赛演示的阅读动线。
        """)

# -------------------- 标签页 3：模型指标 --------------------
with tab_metric:
    st.markdown("以下指标来自测试集评估结果（YOLO11n-Pose，CPU 训练）。")

    box_metrics = pd.DataFrame([
        ["检测框 Precision", "0.99671"],
        ["检测框 Recall", "1.00000"],
        ["检测框 mAP50", "0.99500"],
        ["检测框 mAP50-95", "0.80190"],
    ], columns=["指标", "数值"])
    pose_metrics = pd.DataFrame([
        ["关键点 Precision", "0.97300"],
        ["关键点 Recall", "0.97620"],
        ["关键点 mAP50", "0.96250"],
        ["关键点 mAP50-95", "0.92400"],
    ], columns=["指标", "数值"])

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**检测框（Box）指标**")
        st.dataframe(box_metrics, use_container_width=True, hide_index=True)
    with m2:
        st.markdown("**关键点（Pose）指标**")
        st.dataframe(pose_metrics, use_container_width=True, hide_index=True)

    st.markdown("**数据集划分**")
    split_df = pd.DataFrame([
        ["采集原始图像", 450, "真实稻穗照片"],
        ["清洗后有效样本", 438, "剔除空标、重复与损坏样本"],
        ["训练集", 263, "模型拟合"],
        ["验证集", 88, "训练过程监控与选权重"],
        ["测试集", 87, "未参与训练，用于指标评估与泛化检查"],
    ], columns=["数据集", "样本数", "说明"])
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.markdown("**训练损失（末轮）**")
    loss_df = pd.DataFrame([
        ["train/box_loss", "0.92381"],
        ["train/pose_loss", "0.36121"],
        ["val/box_loss", "1.11667"],
        ["val/pose_loss", "0.36433"],
    ], columns=["损失项", "数值"])
    st.dataframe(loss_df, use_container_width=True, hide_index=True)

    st.info("ℹ️ 说明：上表为检测与关键点定位精度；穗长厘米级测量与人工实测的偏差（MAE/RMSE）"
            "需在补充标尺标定样本后单独评估，当前版本不预作结论。")

# -------------------- 标签页 4：使用说明 --------------------
with tab_guide:
    st.markdown("""
### 📖 操作步骤
1. **启动应用**：在应用目录下执行 `streamlit run app.py`，浏览器自动打开页面。
2. **模型与标定**：确认侧边栏 `best.pt` 路径正确；按拍摄标尺核对标定系数（cm/px）。
3. **上传图片**：选择本地稻穗照片，或勾选内置示例图；背景平整、穗与茎秆完整入镜。
4. **开始分析**：调用你自己的测穗长脚本（`05_infer_curve.py`）检测稻穗、定位穗颈节与穗尖、追踪主穗轴并输出穗长。
5. **查看与导出**：核对结果图与明细表，可下载 CSV 数据与结果图用于材料整理。

### 📏 标定方法
- 在拍摄画面中放入直尺或已知长度的标尺，量出其对应的像素长度；
- `标定系数 = 实际厘米数 ÷ 对应像素数`（例如 5cm 对应 547px → 0.009141 cm/px）；
- 更换机位、焦距、拍摄距离后必须重新标定，否则厘米尺寸整体失真。

### ⚠️ 注意事项
- 穗子相互遮挡、穗尖被裁切、背景杂乱时，关键点与穗轴追踪可能偏差，需人工复核。
- 结果图上的标记尺寸会随原图分辨率放大（4K 原图缩到网页会变很小），若仍看不清，可在侧边栏把「关键点与线条大小倍数」调大。
- 结果区提供「🔎 关键点局部放大」，可逐穗核对蓝点（穗颈节）与绿点（穗尖）是否落在正确位置。
- 黄色曲线为沿弯曲穗轴的曲线长度；直线穗长为穗颈节—穗尖两点距离，仅作复核基线。
- 曲线穗长算法仍在迭代，正式科研比对请以人工实测口径为准。
- 若某穗曲线长度明显异常偏长，多为弧线偏移到侧枝梗，请结合结果图人工判断。
""")

st.divider()
st.caption("水稻穗长智能测量与最长穗筛选系统 · 比赛演示版本 · 面向农科院育种研究与智慧农业推广")
