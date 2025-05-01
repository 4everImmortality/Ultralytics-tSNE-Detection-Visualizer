# 确保这些类文件在正确的位置（例如 util/）并且可以被导入
from tsne.DirectoryProcessor import DirectoryProcessor
from tsne.Yolo11Visualizer import Yolo11Visualizer  # 使用修改后的版本

# 定义参数
PERPLEXITY = 30  # 可以根据您的数据集大小调整
COLORMAP = "hsv"    # 或选择其他 colormap
IMAGE_DIR = r""  # 原始图片文件夹
MODEL_PATH = r""  # YOLO 检测模型路径
WORKERS = 16       # 根据您的CPU核心数调整

# 用于输入到检测模型的图片尺寸
IMAGE_SIZE_FOR_DETECTION = (640, 640)  # (height, width)

# 用于提取特征时，裁剪出的对象小图将被 resize 到的尺寸
# 通常选择一个分类模型常用的输入尺寸，如 ResNet 的 224x224
OBJECT_IMAGE_SIZE_FOR_FEATURES = (224, 224)  # (height, width)

# 检测阈值
CONFIDENCE_THRESHOLD = 0.5  # 对象置信度阈值
IOU_THRESHOLD = 0.7       # 非极大值抑制 (NMS) if NMS possible like yolov8 的 IOU 阈值

# --- 设置要提取对象特征的目标层名称 ---
# 您需要查看模型结构打印输出，选择一个您感兴趣的、适合用于对象特征提取的层。
# 这通常是 backbone 的最后一层，或者 neck 的某个输出层，在其后接 GAP 应该能得到不错的对象特征。
# 假设您选择了一个名为 "model.10.cv2.2.conv" 的层作为示例 (请根据实际模型结构选择)
# <--- **** 修改这里为您实际的目标层名称 ****
TARGET_LAYER_FOR_OBJECT_FEATURES = "model.model.9"


# 1. 获取原始图像文件路径列表
print(f"Scanning images in: {IMAGE_DIR}")
image_paths = DirectoryProcessor.get_all_files(IMAGE_DIR, include_sub_dir=True)
print(f"Found {len(image_paths)} images.")

if not image_paths:
    print("No images found. Exiting.")
    exit()

# 2. 加载 YOLO 检测模型
print(f"Loading model from: {MODEL_PATH}")
try:
    visualizer = Yolo11Visualizer(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 设置用于对象特征提取的目标层 ---
try:
    visualizer.set_target_layer(TARGET_LAYER_FOR_OBJECT_FEATURES)
except ValueError as e:
    print(f"Error setting target layer: {e}")
    print("Please check the model architecture printed above and provide a correct layer name.")
    exit()


# 3. 执行检测，提取对象特征，并计算 t-SNE
# calculate_tsne 现在执行检测和特征提取 for EACH DETECTED OBJECT
print(
    f"Performing detection and calculating t-SNE for objects from layer '{TARGET_LAYER_FOR_OBJECT_FEATURES}' with perplexity={PERPLEXITY}...")
print(
    f"Detection Params: Img Size={IMAGE_SIZE_FOR_DETECTION}, Conf Thresh={CONFIDENCE_THRESHOLD}, IOU Thresh={IOU_THRESHOLD}")
print(
    f"Object Feature Extraction Params: Object Img Size={OBJECT_IMAGE_SIZE_FOR_FEATURES}")

try:
    visualizer.calculate_tsne(
        image_paths,  # 输入是原始图片路径
        perplexity=PERPLEXITY,
        imgsz=IMAGE_SIZE_FOR_DETECTION,       # 用于检测时输入的图片尺寸
        object_imgsz=OBJECT_IMAGE_SIZE_FOR_FEATURES,  # 用于对象特征提取时输入的图片尺寸
        score_threshold=CONFIDENCE_THRESHOLD,  # 检测置信度阈值
        iou_threshold=IOU_THRESHOLD,         # NMS 阈值
        worker=WORKERS
    )
    print("Object t-SNE calculation complete.")
except Exception as e:
    print(f"Error during object t-SNE calculation: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 检查是否有结果数据 (检测到的对象)
if not visualizer.cls_pts:
    print("No objects were detected or processed for plotting. Exiting.")
    exit()

# 4. 绘制并保存完整的 t-SNE 可视化图 (按检测类别区分)
print("Plotting full t-SNE scatter plot of detected objects...")
try:
    visualizer.plot_tsne(
        colormap_name=COLORMAP,
        # 更新标题以反映是对象t-SNE，以及使用的层和尺寸
        title=f"t-SNE (Objects from {TARGET_LAYER_FOR_OBJECT_FEATURES}) Perplexity={PERPLEXITY}",
        export=True  # 将图和数据保存到 result_objects 文件夹
    )
    print("Full object plot displayed and exported.")
except Exception as e:
    print(f"Error plotting full object t-SNE: {e}")
    import traceback
    traceback.print_exc()


print("Script finished.")
