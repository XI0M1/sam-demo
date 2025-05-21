from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import torch

# 使用相对 config 文件路径
config_file = "configs/sam2.1/sam2.1_hiera_s.yaml"
checkpoint = "checkpoints/sam2.1_hiera_small.pt"

# 让 build_sam2 自动处理 compose 和 instantiate
model = build_sam2(config_file=config_file, ckpt_path=checkpoint)
predictor = SAM2ImagePredictor(model)

image = cv2.imread("input/dog1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    input_points = np.array([[500, 375]])
    input_labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

print(scores)
