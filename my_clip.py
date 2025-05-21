import sys
import os
import cv2
sys.path.append(os.path.abspath("segment_anything"))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import clip
import torch
import numpy as np
from PIL import Image

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# # 加载 CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device,download_root="checkpoints")

uploaded_file = "input/dog1.jpg"  # 替换为实际路径
image = Image.open(uploaded_file).convert("RGB")
np_image = np.array(image)

# Step 1: 生成所有掩码
masks = mask_generator.generate(np_image)
# for i, mask_data in enumerate(masks):
#     mask = mask_data["segmentation"]
#     cv2.imwrite(f"output/mask_{i}.png", mask.astype(np.uint8) * 255)  # 保存为二值图

# Step 2: 提取每个掩码区域图像 (使用掩码过滤背景)
patches = []
# masks 是 SamAutomaticMaskGenerator 生成的列表
for mask_data in masks:
    mask = mask_data["segmentation"]
    bbox = mask_data["bbox"]
    x, y, w, h = map(int, bbox)

    # 创建一个黑色背景图像，大小与原图相同
    # masked_image = np.zeros_like(np_image) # 保持3通道

    # 或者，使用原图作为基础，然后用掩码选择区域
    # 注意：这里直接在原图的副本上操作更简单
    masked_patch_img = np_image.copy()

    # 将掩码外的像素设置为黑色
    # mask 是 bool 类型的数组 (True 表示前景)
    # ~mask 是 bool 类型的数组 (True 表示背景)
    masked_patch_img[~mask] = 127  # 将掩码外的像素设置为黑色

    # 现在裁剪 bounding box 区域，但裁剪的是已经背景清空的图像
    # 这样做确保即使 bounding box 较大，送入 CLIP 的也是掩码区域内的内容（和黑背景）
    patch = masked_patch_img[y:y + h, x:x + w]

    # 将 numpy 数组转换为 PIL Image
    # 检查 patch 是否为空，或者尺寸是否有效
    if patch.shape[0] > 0 and patch.shape[1] > 0:
        patch_pil = Image.fromarray(patch, 'RGB')
        # 对裁剪后的图像块进行 CLIP 预处理
        patches.append(preprocess(patch_pil).unsqueeze(0).to(device))
    else:
        # 如果 patch 无效，跳过
        print(f"Warning: Skipping invalid patch with bbox {bbox}")

# 检查是否生成了有效的 patches
if not patches:
    print("Error: No valid patches were generated from the masks.")
    # 这里可能需要添加错误处理逻辑，例如退出或返回空结果
    # 假设下面使用 patches 的代码块只会在 patches 非空时执行
    similarity = None  # 标记没有相似度结果

else:
    # 确保这里的 image_features 使用了处理过的 patches
    image_features = torch.cat(patches)

    # Step 3: 计算文本与每个区域相似度
    text = clip.tokenize(["a dog with white skin", "a dog with white and brown skin"]).to(device)  # 你输入的文字，可以尝试更具体的描述

    # 确保 CLIP 模型已加载且正确
    if 'clip_model' not in locals() or clip_model is None:
        # 如果 CLIP 未加载（因为原始代码注释了），则加载
        print("Loading CLIP model...")
        clip_model, preprocess = clip.load("ViT-B/32", device=device, download_root="checkpoints")

    # 确保 CLIP 模型已加载且正确
    if 'clip_model' not in locals() or clip_model is None:
        raise RuntimeError("CLIP model is not loaded.")

    # 获取特征
    with torch.no_grad():  # 在评估模式下不需要梯度计算
        image_features = clip_model.encode_image(image_features)
        text_features = clip_model.encode_text(text)

    # 规范化特征（CLIP 默认会返回已规范化的特征，但为了稳妥，可以再规范化一次）
    # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算相似度 (点积等价于规范化向量的余弦相似度)
    similarity = image_features @ text_features.T  # shape: [num_patches, num_texts]

    # Step 4: 选出最佳匹配掩码
    # 找到每个文本描述最匹配的掩码索引
    best_mask_indices_per_text = similarity.argmax(dim=0)  # shape: [num_texts]

    # 如果你想找到 overall 最匹配的 (即所有patches vs 所有texts 中最高的那个)
    # overall_best_score, overall_best_idx_flat = similarity.max(dim=None)
    # 移除命名维度，然后找到所有元素中的最大值和索引
    overall_best_score, overall_best_idx_flat = similarity.flatten().max(dim=0)
    num_patches = similarity.shape[0]
    overall_best_mask_idx = overall_best_idx_flat // similarity.shape[1]
    overall_best_text_idx = overall_best_idx_flat % similarity.shape[1]

    labels = ["a dog with white skin", "a dog with white and brown skin"]  # 与 clip.tokenize 中的文本对应

    print(f"Similarity shape: {similarity.shape}")  # 应该是 [num_patches, num_texts]

    print("\nBest mask index for each text prompt:")
    for i, text_label in enumerate(labels):
        best_mask_idx_for_this_text = best_mask_indices_per_text[i].item()
        # 打印对应的掩码索引
        print(f"  Prompt '{text_label}': Mask Index {best_mask_idx_for_this_text}")
        # 你可以选择可视化这个特定的掩码

    print(f"\nOverall best match:")
    print(f"  Best Mask Index: {overall_best_mask_idx.item()}")
    print(f"  Best Text Prompt: '{labels[overall_best_text_idx.item()]}'")
    print(f"  Similarity Score: {overall_best_score.item():.4f}")

    # Step 5: 保存整体最佳匹配的掩码可视化
    if overall_best_mask_idx is not None:
        best_mask = masks[overall_best_mask_idx.item()]["segmentation"]

        import cv2

        # 将最佳掩码可视化叠加在原图上
        vis_image = np_image.copy()

        # 创建一个叠加层，只在掩码区域有颜色
        overlay = np.zeros_like(vis_image, dtype=np.uint8)
        overlay[best_mask] = [0, 255, 0]  # 掩码区域上色为绿色 (BGR 格式)

        # 将叠加层与原图融合
        alpha = 0.5  # 透明度
        vis_image = cv2.addWeighted(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR), 1 - alpha, overlay, alpha, 0)

        # vis_image[best_mask] = [0, 255, 0] # 简单粗暴的上色 (BGR)

        cv2.imwrite("output/masked_result_overall_best.png", vis_image)
        print("Overall best matching masked image saved as output/masked_result_overall_best.png")

        # 你也可以选择保存每个文本prompt对应的最佳掩码
        # for i, text_label in enumerate(labels):
        #      best_mask_idx_for_this_text = best_mask_indices_per_text[i].item()
        #      mask_for_vis = masks[best_mask_idx_for_this_text]["segmentation"]
        #      vis_image_single = np_image.copy()
        #      overlay_single = np.zeros_like(vis_image_single, dtype=np.uint8)
        #      overlay_single[mask_for_vis] = [0, 0, 255] # 例如，红色
        #      vis_image_single = cv2.addWeighted(cv2.cvtColor(vis_image_single, cv2.COLOR_RGB2BGR), 0.7, overlay_single, 0.3, 0)
        #      # 清理标签字符串中的非法字符以便文件名使用
        #      safe_label = "".join([c for c in text_label if c.isalnum() or c in (' ', '-', '_')]).replace(' ', '_')
        #      cv2.imwrite(f"output/masked_result_{safe_label}.png", vis_image_single)
        #      print(f"Masked image for '{text_label}' saved as output/masked_result_{safe_label}.png")

    else:
        print("No valid mask found to save visualization.")


