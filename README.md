# YOLOv11 t-SNE Visualizer

## 简介 (Introduction)

本项目用于将 YOLOv11 检测模型的特征通过 t-SNE 降维并可视化，帮助分析模型在不同类别、不同对象上的特征分布。支持对每个检测到的对象进行特征提取和可视化。

This project visualizes features from a YOLOv11 detection model using t-SNE, helping you analyze feature distributions across different classes and objects. It supports per-object feature extraction and visualization.

---

## 主要功能 (Features)

- 支持 YOLOv11 检测模型的特征提取  
  Feature extraction from YOLOv11 detection models
- 对每个检测到的对象进行特征降维与可视化  
  Per-object feature dimensionality reduction and visualization
- 支持多类别对比可视化  
  Multi-class comparison visualization
- 支持多线程加速处理  
  Multi-threaded processing

---

## 快速开始 (Quick Start)

1. **安装依赖 (Install requirements):**

   ```sh
   pip install -r requirements.txt
   ```

2. **准备模型和图片 (Prepare model and images):**

- 将你的 YOLOv11 检测模型权重放在合适路径
- 准备好待分析的图片文件夹
- Place your YOLOv11 model weights and prepare your image folder.

运行主程序 (Run main script):

3. **运行主程序 (Run main script):**
   ```python
   python Visualizer.py
   ```
   按照脚本中的参数说明修改模型路径、图片路径、目标层名称等。

   Modify model path, image path, and target layer name in the script as needed.
4. **主要参数说明 (Main Parameters)**
   **IMAGE_DIR**：图片文件夹路径 (Image folder path)
   **MODEL_PATH**：YOLOv11 检测模型权重路径 (YOLOv11 model weights path)
   **TARGET_LAYER_FOR_OBJECT_FEATURES**：用于特征提取的目标层名称 (Target layer name for feature extraction)
   **PERPLEXITY**：t-SNE 的 perplexity 参数 (t-SNE perplexity)
   **COLORMAP**：可视化的颜色映射 (Colormap for visualization)
5. **结果输出 (Results)**
- 可视化结果图片和数据将保存在 result_objects/ 和 compare_objects/ 文件夹下。
- Visualization images and data will be saved in result_objects/ and compare_objects/ folders.
6. **依赖 (Requirements)**
   见下方 requirements.txt
   See requirements.txt below.

7. **目录结构 (Directory Structure)**
```
YOLOv11-tSNE-Visualizer-main/
│
├── Visualizer.py
├── requirements.txt
├── README.md
└── tsne/
    ├── DirectoryProcessor.py
    ├── TextProgressBar.py
    └── Yolo11Visualizer.py
```

**如有问题欢迎提 issue。**
**Feel free to open an issue if you have any questions.**