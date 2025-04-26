import sys
import os
sys.path.append(os.path.abspath("segment_anything"))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# 初始化模型
@st.cache_resource
def load_predictor():
    sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

predictor = load_predictor()

# 上传图片
st.title("📌 SAM 本地 Demo")
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="原始图像")

    x = st.number_input("X坐标", 0, image_np.shape[1]-1, value=image_np.shape[1]//2)
    y = st.number_input("Y坐标", 0, image_np.shape[0]-1, value=image_np.shape[0]//2)

    if st.button("生成掩码"):
        predictor.set_image(image_np)
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # foreground

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]
        vis = image_np.copy()
        vis[mask] = [0, 255, 0]  # 掩码区域着色为绿色

        st.image(vis, caption="掩码结果")