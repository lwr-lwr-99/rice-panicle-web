# 水稻穗长智能测量与最长穗筛选系统（Web 部署版）

基于 YOLOv11n-Pose（关键点检测）+ 主穗轴平滑弧线提取的水稻穗长测量系统。
本仓库可直接部署到 Streamlit Community Cloud（免费），部署后得到一个公网链接，
手机浏览器打开即可上传拍照、看结果、导出 CSV。

## 一、部署步骤（约 10 分钟）

1. 打开 https://share.streamlit.io ，用 GitHub 账号登录并授权
2. 点 **Create app**，填写：
   - Repository：`lwr-lwr-99/rice-panicle-web`
      - Branch：`main`
         - **Main file path：`app.py`**（本仓库是扁平结构，不要再写 panicle_app/app.py）
         3. 点 **Deploy**，首次要装 torch/ultralytics，约 5–10 分钟
         4. 拿到链接形如 `https://xxx.streamlit.app`，发到微信文件传输助手，手机点开即用

         ## 二、文件说明

         | 文件 | 作用 |
         |---|---|
         | `app.py` | Streamlit 入口（主程序） |
         | `pipeline.py` | 内置测长管线与可视化绘制 |
         | `legacy_curve.py` | 桥接原始测穗长脚本 |
         | `05_infer_curve.py` | 原始测穗长算法（网页直接调用，不改写） |
         | `best.pt` | YOLOv11n-Pose 权重（5.36MB） |
         | `simsun.ttc` | 中文字体（已子集化，0.17MB，云端显示中文用） |
         | `sample.jpg` | 内置示例图 |
         | `requirements.txt` | Python 依赖 |
         | `packages.txt` | 系统库 libgl1 / libglib2.0-0 |

         ## 三、手机使用

         打开链接 → 若提示“唤醒应用”点一下，等 30–60 秒冷启动 → 上传/拍照 →
         填标定系数（默认 0.009141 cm/px）→ 开始分析 → 表格与 CSV 可导出。

         ## 四、指标口径（与申报书一致）

         - 检测框：mAP50 99.50% / mAP50-95 69.19% / P 99.67% / R 100%
         - 关键点：mAP50 96.25% / mAP50-95 92.40% / P 97.30% / R 97.62%（OKS-based）
         - 数据集 438 张（263/88/87），训练 50 轮，模型 5.36MB

         
         - 
