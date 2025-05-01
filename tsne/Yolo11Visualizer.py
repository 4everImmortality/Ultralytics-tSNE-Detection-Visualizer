import cv2
import os
import threading
import torch
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple
from sklearn.manifold import TSNE
from ultralytics import YOLO
from matplotlib import colormaps
from queue import Queue
from itertools import combinations

from tsne.TextProgressBar import TextProgressBar

class Yolo11Visualizer:
    def __init__(self, model:str, seed:int=42) -> None:
        self.__model = YOLO(model, task='detect') # 确保是 detect 任务
        # 初始化模型，使用一个标准尺寸输入
        _ = self.__model(torch.zeros(1, 3, 640, 640), verbose=False)
        self.__seed = seed
        self.__cls_pts = None
        self.__target_layer_name = None # 目标层名称，用于对象特征提取
        self._hook_output = None # 用于存储钩子捕获的输出
        self.__class_names = self.__model.names # 获取模型检测的类别名称列表

        print("\n--- Model Architecture (for object feature extraction) ---")
        # 打印模型结构，帮助用户选择用于对象特征提取的层
        for name, module in self.__model.named_modules():
             if len(list(module.children())) > 0 or name == '':
                 print(f"Layer Name: {name}, Module Type: {type(module).__name__}")
        print("-----------------------------------------------------\n")


    def set_target_layer(self, layer_name: str) -> None:
        """
        Sets the name of the layer from which to extract features for detected objects.
        Use the printed model architecture to find layer names.
        """
        found = False
        for name, _ in self.__model.named_modules():
            if name == layer_name:
                found = True
                break
        if not found:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

        self.__target_layer_name = layer_name
        print(f"Target layer for object feature extraction set to: {layer_name}")


    # 新方法：提取单个对象（裁剪后的小图）的特征
    def _extract_object_features(self, object_image: np.ndarray) -> torch.Tensor:
         if self.__target_layer_name is None:
             raise ValueError("Target layer name for object feature extraction is not set. Call set_target_layer() first.")

         def hook_fn(module, input, output):
             self._hook_output = output.detach().cpu()

         hook_handle = None
         for name, module in self.__model.named_modules():
             if name == self.__target_layer_name:
                 hook_handle = module.register_forward_hook(hook_fn)
                 break

         if hook_handle is None:
              raise ValueError(f"Could not find layer '{self.__target_layer_name}' to register hook for object feature extraction.")

         try:
              # 输入图像应该是裁剪并 resize 后的对象小图，形状为 [H, W, C]
              # 转换为模型所需的 [N, C, H, W]，归一化
              image_tensor = torch.from_numpy(object_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
              if torch.cuda.is_available():
                  image_tensor = image_tensor.cuda()

              # 在模型上运行前向传播，提取对象特征
              # 这里的 forward pass 只会计算到 self.__target_layer_name 层
              # 注意：直接调用 self.__model(image_tensor) 会运行整个模型，包括检测头
              # 如果目标层在 backbone/neck，这会 work，但可能有点浪费计算
              # 如果目标层在检测头，可能需要更复杂的调用方式，或者选择 backbone/neck 的层
              # 我们假设目标层在 backbone 或 neck，其输出是特征图 [1, C, H', W']
              self.__model(image_tensor, verbose=False) # 触发钩子

         finally:
             hook_handle.remove() # 移除钩子

         features_tensor = self._hook_output
         self._hook_output = None # 清空存储

         if features_tensor is None:
              raise RuntimeError(f"Hook did not capture output for layer '{self.__target_layer_name}' during object feature extraction.")

         # --- 后处理捕获的特征 ---
         # 假设捕获的特征是 [1, C, H', W'] 或 [1, C]
         if features_tensor.ndim == 4 and features_tensor.shape[0] == 1:
             # 应用全局平均池化到空间维度 [H', W'] -> [1, C]
             pooled_features = torch.mean(features_tensor, dim=(-1, -2))
             # 移除批次维度 -> [C]
             return pooled_features.squeeze(0)

         elif features_tensor.ndim == 2 and features_tensor.shape[0] == 1:
             # 如果已经是 [1, C] 形状，直接移除批次维度 -> [C]
             return features_tensor.squeeze(0)
         else:
              print(f"Error: Captured object feature from layer '{self.__target_layer_name}' has unexpected shape: {features_tensor.shape}")
              raise ValueError(f"Unsupported output shape for layer '{self.__target_layer_name}' during object feature extraction: {features_tensor.shape}")


    # 修改 calculate_tsne 以处理对象实例
    def calculate_tsne(self, image_paths: List[str], perplexity:int=30, imgsz: Tuple[int, int] = (640, 640), object_imgsz: Tuple[int, int] = (224, 224), score_threshold: float = 0.5, iou_threshold: float = 0.45, worker:int=4): # 添加 object_imgsz, score_threshold, iou_threshold
        # --- 检查目标层是否已设置 ---
        if self.__target_layer_name is None:
            raise ValueError("Target layer name for object feature extraction is not set. Call set_target_layer() before calculate_tsne().")
        # --- 检查 imgsz 和 object_imgsz ---
        if imgsz is None or object_imgsz is None:
             print("Warning: imgsz or object_imgsz is not set. Using defaults.") # 使用默认值 (640, 640) 和 (224, 224)

        prog_bar = TextProgressBar(len(image_paths)) # 进度条仍然显示图片处理进度

        # 结果队列存储 (原始图片路径, 检测到的类别名称, 对象特征向量, bounding box)
        result_queue = Queue()
        lock = threading.Lock() # 用于线程安全打印


        def execute_revised(path):
            image = cv2.imread(path)
            if image is None:
                 with lock:
                     print(f"\nWarning: Could not load image {path}. Skipping.")
                 prog_bar.add_step()
                 return

            original_h, original_w = image.shape[:2] # 记录原始尺寸

            # --- 1. 对原图进行检测 ---
            # 首先将原图 resize 到 imgsz 输入模型进行检测
            # YOLO 模型通常需要 NCHW 格式的张量输入，归一化
            detect_input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 检测通常用 RGB
            if imgsz is not None:
                 try:
                     detect_input_image = cv2.resize(detect_input_image, (imgsz[1], imgsz[0]))
                 except Exception as resize_e:
                     with lock:
                         print(f"\nWarning: Could not resize image {path} for detection to {imgsz}. Skipping. Error: {resize_e}")
                     prog_bar.add_step()
                     return

            detect_input_tensor = torch.from_numpy(detect_input_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            if torch.cuda.is_available():
                detect_input_tensor = detect_input_tensor.cuda()

            try:
                 # 运行检测
                 results = self.__model(detect_input_tensor, conf=score_threshold, iou=iou_threshold, verbose=False)
                 # results 是一个 Results 对象列表，通常只有一个元素 results[0]
                 if not results or not results[0].boxes or len(results[0].boxes) == 0:
                     # with lock: # 如果没有检测到任何对象，也打印提示 (可选)
                     #    print(f"\nNo objects detected in {path}.")
                     prog_bar.add_step()
                     return # 如果没有检测到对象，跳过特征提取和结果添加

                 # --- 2. 处理检测结果并提取每个对象的特征 ---
                 detected_boxes = results[0].boxes.xyxy # Get boxes in xyxy format
                 detected_scores = results[0].boxes.conf # Get scores
                 detected_classes = results[0].boxes.cls # Get class IDs

                 for i in range(len(detected_boxes)):
                     bbox_xyxy = detected_boxes[i].tolist() # Bounding box coordinates [x1, y1, x2, y2] in original image scale if from results[0].boxes.xyxy
                     score = detected_scores[i].item()
                     class_id = int(detected_classes[i].item())

                     # Get class name
                     if class_id < len(self.__class_names):
                         class_name = self.__class_names[class_id]
                     else:
                         class_name = f"unknown_class_{class_id}" # Handle unknown classes

                     # --- Crop the object ---
                     # Crop from the ORIGINAL image using scaled coordinates if needed
                     # Note: results[0].boxes.xyxy usually returns coordinates scaled to original image size
                     x1, y1, x2, y2 = map(int, bbox_xyxy)
                     # Ensure coordinates are within image bounds
                     x1 = max(0, x1)
                     y1 = max(0, y1)
                     x2 = min(original_w, x2)
                     y2 = min(original_h, y2)

                     if x2 <= x1 or y2 <= y1: # Skip invalid boxes
                         continue

                     cropped_object = image[y1:y2, x1:x2] # Crop from original BGR image

                     if cropped_object.size == 0: # Skip empty crops
                         continue

                     # --- Resize the cropped object ---
                     if object_imgsz is not None:
                         try:
                              # Resize cropped object to object_imgsz for feature extraction
                             resized_cropped_object = cv2.resize(cropped_object, (object_imgsz[1], object_imgsz[0])) # cv2 resize expects (width, height)
                             # Convert to RGB for consistency if _extract_object_features expects RGB
                             resized_cropped_object = cv2.cvtColor(resized_cropped_object, cv2.COLOR_BGR2RGB)
                         except Exception as obj_resize_e:
                             with lock:
                                 print(f"\nWarning: Could not resize cropped object from {path} ({bbox_xyxy}) to {object_imgsz}. Skipping object. Error: {obj_resize_e}")
                             continue # Skip this object
                     else:
                          # If no object_imgsz, maybe process original crop?
                          # But _extract_object_features expects a fixed size input for consistent features
                          # So object_imgsz is effectively required for method A.
                          with lock:
                              print("\nError: object_imgsz is not set, but required for cropping and feature extraction.")
                          raise ValueError("object_imgsz must be set for object instance visualization.")


                     # --- 3. 提取对象特征 ---
                     try:
                         object_features = self._extract_object_features(resized_cropped_object) # Call the new method
                         # Add result to the queue: (原始图片路径, 检测类别名称, 对象特征, bounding box)
                         result_queue.put((path, class_name, object_features, bbox_xyxy))
                     except Exception as obj_extract_e:
                         with lock:
                             print(f"\nWarning: Could not extract features for object ({class_name}, {bbox_xyxy}) from {path}. Skipping object. Error: {obj_extract_e}")
                         pass # Allow thread to finish even on object extraction error

            except Exception as detect_e:
                with lock:
                    print(f"\nError during detection or processing results for image {path}. Skipping image. Error: {detect_e}")
                pass # Allow thread to finish even on image processing error

            # --- Increment Progress Bar ---
            # 进度条按处理的图片数量推进
            prog_bar.add_step()


        # --- Main Thread Management ---
        print(f"\nProcessing images, performing detection, and extracting object features (Target Layer: {self.__target_layer_name})...")
        prog_bar = TextProgressBar(len(image_paths))
        thread_queue = Queue()

        for path in image_paths:
             while thread_queue.qsize() >= worker:
                 oldest_thread = thread_queue.get()
                 oldest_thread.join()

             thread = threading.Thread(target=execute_revised, args=(path,), daemon=True)
             thread.start()
             thread_queue.put(thread)

        # Wait for all processing threads to complete
        while not thread_queue.empty():
             thread = thread_queue.get()
             thread.join()

        # Collect results from the result_queue
        processed_object_data = []
        while not result_queue.empty():
             processed_object_data.append(result_queue.get())

        # --- Prepare data for t-SNE (Each point is a detected object) ---
        if not processed_object_data:
            print("\nError: No valid objects were detected or processed from any image. Cannot perform t-SNE.")
            self.__cls_pts = []
            return

        # processed_object_data is a list of (original_path, class_name, feature_vector[C], bbox) tuples
        # t-SNE needs features [N_objects, C] and labels [N_objects]
        object_paths = [item[0] for item in processed_object_data] # Original image paths for context
        object_class_names = [item[1] for item in processed_object_data] # Detected class names (labels for t-SNE)
        object_features_tensors = [item[2] for item in processed_object_data] # List of [C] tensors
        object_bboxes = [item[3] for item in processed_object_data] # Bounding boxes for context

        # Check if all feature tensors have the same size before stacking
        if not object_features_tensors:
            print("\nError: No object feature tensors collected.")
            self.__cls_pts = []
            return

        first_shape = object_features_tensors[0].shape
        if not all(f.shape == first_shape for f in object_features_tensors):
            print("\nError: Object feature tensors have inconsistent shapes after extraction and pooling.")
            # Print info for debugging
            for i, f in enumerate(object_features_tensors):
                if f.shape != first_shape:
                    print(f"Object at index {i} (Image: {object_paths[i]}, Class: {object_class_names[i]}, Bbox: {object_bboxes[i]}) has shape {f.shape}, expected {first_shape}")
            raise ValueError("Inconsistent object feature shapes.")

        # Stack the [C] tensors -> [Num_objects, C]
        features_for_tsne = torch.stack(object_features_tensors).cpu().numpy()

        # --- Perform t-SNE reduction ---
        print(f"\nPerforming t-SNE reduction on {features_for_tsne.shape[0]} detected objects with {features_for_tsne.shape[1]} features...")
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.__seed, n_jobs=-1)
            reduced_features = tsne.fit_transform(features_for_tsne)
            print("t-SNE reduction complete.")
        except Exception as tsne_e:
             print(f"\nError during t-SNE fit_transform: {tsne_e}")
             if np.isnan(features_for_tsne).any() or np.isinf(features_for_tsne).any():
                  print("Warning: Feature data contains NaN or Inf values, which can cause t-SNE to fail.")
             raise tsne_e


        # --- Bind data (Each point is a detected object instance) ---
        self.__cls_pts = []
        for i in range(len(reduced_features)):
            # Store (原始图片路径, 检测到的类别名称, t-SNE坐标, bounding box)
            self.__cls_pts.append((
                object_paths[i],
                object_class_names[i],
                tuple(map(float, reduced_features[i])),
                object_bboxes[i] # Store bbox for potential future use (e.g., clicking on point)
            ))

        print("COMPLETE")
        print(f"Collected {len(self.__cls_pts)} data points (detected objects).")
        print("call property 'cls_pts' to retrieve object data points")


    # plot_tsne 和 plot_compare_tsne 需要使用新的数据结构 __cls_pts
    # __cls_pts 现在是 (原始图片路径, 检测类别名称, t-SNE坐标, bounding box) 的列表

    def plot_tsne(self, colormap_name: str, title: str="t-SNE Scatter (Detected Objects)", export: bool = False): # Update default title
        # Validate colormap
        if colormap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap '{colormap_name}'. Available colormaps: {plt.colormaps()}")

        # Convert data into dictionary with DETECTED class names as keys
        class_points = {} # Key is detected class name, Value is list of (coordinates, original_path, bbox)
        if self.__cls_pts:
            for original_path, detected_class_name, coordinates, bbox in self.__cls_pts:
                class_points.setdefault(detected_class_name, []).append((coordinates, original_path, bbox))
        else:
             print("No detected object data to plot.")
             return

        # Determine number of unique detected classes
        detected_classes = list(class_points.keys())
        num_classes = len(detected_classes)
        if num_classes == 0:
            print("No detected classes found in data to plot.")
            return

        cmap = colormaps[colormap_name]

        fig = plt.figure(figsize=(10, 7))
        # Use sorted detected class names for consistent color mapping
        sorted_detected_classes = sorted(detected_classes)

        for idx, class_name in enumerate(sorted_detected_classes):
             points_data = class_points[class_name] # List of (coordinates, original_path, bbox)
             points = np.array([p[0] for p in points_data]) # Extract just the coordinates

             # Plot points for this detected class
             plt.scatter(points[:, 0], points[:, 1], label=class_name, color=cmap(idx / (num_classes - 1 if num_classes > 1 else 1)), s=20, alpha=0.7) # Smaller point size

        plt.legend(title="Detected Classes", loc='upper left', bbox_to_anchor=(1.05, 1)) # Update legend title
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        if export:
            os.makedirs("result_objects", exist_ok=True) # Export to a different folder
            # Sanitize title for filename
            safe_title = "".join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title]).strip()
            image_path = os.path.join("result_objects", f"{safe_title}.jpg")
            text_path = os.path.join("result_objects", f"{safe_title}.txt")
            fig.savefig(image_path, format='jpeg', dpi=150)

            if self.__cls_pts: # Check if data exists before writing
                with open(text_path, "w") as file:
                    # Sort data for consistent output
                    # Sort by detected class name, then by t-SNE coords
                    sorted_data = sorted(self.__cls_pts, key=lambda x: (x[1], x[2][0], x[2][1]))
                    for original_path, detected_class_name, coords, bbox in sorted_data:
                        file.write(f"{coords} | {detected_class_name} | {original_path} | {bbox}\n") # Include original path and bbox
            else:
                 print("No object data to export to text file.")


    def plot_compare_tsne(self, colormap_name:str="rainbow", num_classes:int=2, export:bool=False):
        # Validate num_classes
        if num_classes < 2:
            raise ValueError("num_classes should be >= 2")

        # Validate colormap
        if colormap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap '{colormap_name}'. Available colormaps: {plt.colormaps()}")

        # Convert data into dictionary with DETECTED class names as keys
        class_points = {} # Key is detected class name, Value is list of (coordinates, original_path, bbox)
        if self.__cls_pts:
            for original_path, detected_class_name, coordinates, bbox in self.__cls_pts:
                class_points.setdefault(detected_class_name, []).append((coordinates, original_path, bbox))
        else:
             print("No detected object data to plot comparison.")
             return


        detected_classes = list(class_points.keys())
        if len(detected_classes) < num_classes:
             print(f"Warning: Not enough detected classes ({len(detected_classes)}) to form combinations of {num_classes}. Skipping comparison plots.")
             return

        # Generate class combinations using DETECTED class names
        class_combinations = list(combinations(detected_classes, num_classes))
        cmap = colormaps[colormap_name]

        for class_pair in class_combinations:
            fig = plt.figure(figsize=(10, 7))
            title = " vs ".join(list(class_pair))
            export_data = [] # Data for the current pair of classes

            # Use sorted class_pair for consistent color mapping
            sorted_class_pair = sorted(list(class_pair))

            has_points_in_pair = False # Flag to check if any points exist for this pair
            for idx, class_name in enumerate(sorted_class_pair):
                if class_name in class_points:
                    points_data = class_points[class_name] # List of (coordinates, original_path, bbox)
                    points = np.array([p[0] for p in points_data]) # Extract coordinates
                    if points.shape[0] > 0: # Check if there are points for this class
                         has_points_in_pair = True
                         export_data.extend([
                            (original_path, detected_class_name, coords, bbox)
                            for coords, original_path, bbox in points_data
                         ])
                         # Normalize idx based on the number of classes in the current comparison (num_classes)
                         plt.scatter(points[:, 0], points[:, 1], label=class_name, color=cmap(idx / (num_classes - 1 if num_classes > 1 else 1)), s=20, alpha=0.7)

            # Only plot if there were points for the classes in the pair
            if has_points_in_pair:
                plt.legend(title="Detected Classes", loc='upper left', bbox_to_anchor=(1.05, 1)) # Update legend title
                plt.title(title.strip())
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

                if export:
                    os.makedirs("compare_objects", exist_ok=True) # Export to a different folder
                    # Sanitize title for filename
                    safe_title = "".join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title]).strip()
                    image_path = os.path.join("compare_objects", f"{safe_title}.jpg")
                    text_path = os.path.join("compare_objects", f"{safe_title}.txt")
                    fig.savefig(image_path, format='jpeg', dpi=150)

                    if export_data:
                        with open(text_path, "w") as file:
                             # Sort data for consistent output
                             sorted_data = sorted(export_data, key=lambda x: (x[1], x[2][0], x[2][1])) # Sort by detected class then t-SNE coords
                             for original_path, detected_class_name, coords, bbox in sorted_data:
                                 file.write(f"{coords} | {detected_class_name} | {original_path} | {bbox}\n") # Include original path and bbox
                    else:
                         print(f"No object data to export to text file for class pair: {title}.")
            else:
                 print(f"No data points found for detected class pair: {title}. Skipping plot and export.")


    @property
    def cls_pts(self):
        return self.__cls_pts