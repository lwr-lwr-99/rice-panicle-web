# 水稻穗长智能测量与最长穗筛选系统（Web 部署版）

基于 YOLOv11n-Pose（关键点检测）+ 主穗轴平滑弧线提取的水稻穗长测量系统。
本仓库为 **可直接部署到 Streamlit Community Cloud（免费）** 的版本，部署完成会得到一个公网链接，
手机浏览器直接打开即可上传拍照、看结果、导出 CSV。

---

## 一、目录结构

```
rice_panicle_web/
├─ panicle_app/
│  ├─ app.py                     # 网页主程序（Streamlit 入口）
│  ├─ pipeline.py                # 内置测长管线（可视化绘制、中文字体）
│  ├─ legacy_curve.py            # 桥接原始脚本 05_infer_curve.py
│  ├─ best.pt                    # 模型权重（5.36MB）
│  ├─ sample.jpg                 # 内置示例图
│  └─ assets/simsun.ttc          # 中文字体（云端 Linux 无微软字体，随仓库带上）
├─ panicle_longest/
│  └─ scripts/05_infer_curve.py  # 原始测穗长算法（网页直接调用，不改写）
├─ requirements.txt              # Python 依赖
├─ packages.txt                  # 系统库（libgl1 / libglib2.0-0）
└─ .streamlit/config.toml        # 上传上限 50MB、关闭统计
```

> 部署时 **Main file path 必须填 `panicle_app/app.py`**。
> 因为 `legacy_curve.py` 默认按"上一级目录 + panicle_longest/scripts"找原始脚本，保持这个层级才不会报“找不到脚本”。

---

## 二、部署步骤（全程约 10 分钟）

### 1. 建 GitHub 仓库
1. 登录 https://github.com → 右上角 `+` → **New repository**
2. 仓库名随意，例如 `rice-panicle-web`；可见性选 **Public**（Streamlit 免费版只支持公开仓库）
3. 不用勾 README / .gitignore，直接 **Create repository**

### 2. 上传文件（网页拖拽最省事）
1. 进入新仓库 → **Add file → Upload files**
2. 把本项目文件夹里的**所有内容**（`panicle_app`、`panicle_longest`、`requirements.txt`、`packages.txt`、`.streamlit`）整体拖进上传框
   - 拖整个文件夹即可，GitHub 会保留目录结构；`best.pt` 5.36MB、`simsun.ttc` 17.47MB 都在单文件 25MB 限制内
3. 底部 Commit changes → 等进度条走完

> 若 `.streamlit` 隐藏目录拖不进去，可以先忽略（不影响运行，只是没有 50MB 上传上限）。
> 也可用 GitHub Desktop 直接拖到本地克隆目录后 Commit & Push，更稳。

### 3. 部署到 Streamlit Cloud
1. 打开 https://share.streamlit.io → **Sign in with GitHub** 授权
2. 点 **Create app**（或 New app）
3. 填写：
   - Repository：`你的用户名/rice-panicle-web`
   - Branch：`main`
   - **Main file path：`panicle_app/app.py`**
   - App URL：可自定义子域名，例如 `rice-panicle`
4. 点 **Deploy**，首次构建要装 torch/ultralytics，约 5–10 分钟，页面会显示 Building → 变成 Running
5. 拿到链接形如 `https://rice-panicle.streamlit.app`

### 4. 手机上用
- 直接把链接发到微信"文件传输助手"，手机点开即可（无需安装 App）
- 首次打开若显示"唤醒应用"，点一下唤醒按钮，等 30–60 秒冷启动
- 手机拍照 → 上传 → 填标定系数（默认 0.009141 cm/px）→ 开始分析

---

## 三、免费额度与注意事项（很重要）

| 项目 | 说明 |
|---|---|
| 内存 | 免费实例约 1GB 内存；torch CPU 版 + 模型加载后约 600–800MB，单张图推理够用，**不要并发多张** |
| 休眠 | 无人访问约 12 小时后自动休眠，再次打开需冷启动（约 30–60 秒），属正常现象 |
| 上传上限 | 默认 200MB；本项目配置为 50MB，手机照片完全够 |
| 隐私 | 仓库是公开的，**不要把真实育种原始数据、未公开的试验图片放进仓库** |
| 改代码 | 在 GitHub 网页上改文件后，Streamlit Cloud 会自动重新构建（Reboot 按钮可强制） |

---

## 四、本地先自测（可选）

```bash
cd rice_panicle_web
pip install -r requirements.txt
streamlit run panicle_app/app.py
```

浏览器打开 http://localhost:8501 ，勾"使用内置示例图"验证能出图、能导出 CSV，再推 GitHub。

---

## 五、常见报错对照

| 现象 | 原因 / 处理 |
|---|---|
| `FileNotFoundError: 找不到原始测穗长脚本` | Main file path 填错（必须 `panicle_app/app.py`），或 `panicle_longest/scripts/05_infer_curve.py` 没上传 |
| `❌ 未找到模型文件` | `best.pt` 没上传到 `panicle_app/` 下 |
| 图上中文变成方框/问号 | `panicle_app/assets/simsun.ttc` 缺失（本项目已内置） |
| `ModuleNotFoundError: cv2` | requirements.txt 里是 `opencv-python-headless`，且 packages.txt 要有 `libgl1`、`libglib2.0-0` |
| 构建失败 `torch` 安装超时 | 点 Reboot 重试一次；仍失败可把 requirements 里加 `--extra-index-url https://download.pytorch.org/whl/cpu` 并用 CPU 版 torch |
| 页面一直 Building | 首次装依赖慢，超过 15 分钟再 Reboot |

---

## 六、指标口径（与申报书一致）

- 检测框：mAP50 99.50% / mAP50-95 69.19% / P 99.67% / R 100%
- 关键点：mAP50 96.25% / mAP50-95 92.40% / P 97.30% / R 97.62%（Ultralytics 的 OKS-based PoseMetrics）
- 数据集 438 张（训练 263 / 验证 88 / 测试 87），训练 50 轮，模型 5.36MB
