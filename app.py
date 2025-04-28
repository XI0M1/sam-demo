import sys
import os
sys.path.append(os.path.abspath("segment_anything"))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from streamlit_drawable_canvas import st_canvas

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

    # st.image(image, caption="原始图像")
    #
    # x = st.number_input("X坐标", 0, image_np.shape[1]-1, value=image_np.shape[1]//2)
    # y = st.number_input("Y坐标", 0, image_np.shape[0]-1, value=image_np.shape[0]//2)
    st.subheader("点击图像选择点进行分割")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # 点击后小红圈提示
        stroke_width=10,
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="point",  # 只允许点
        key="canvas",
    )
    if canvas_result.json_data is not None:
        if len(canvas_result.json_data["objects"]) > 0:
            # 获取最后一个点击的点
            points = []
            # point = canvas_result.json_data["objects"][-1]
            for obj in canvas_result.json_data["objects"]:
                x = int(obj["left"])
                y = int(obj["top"])
                points.append([x, y])
            points = np.array(points)  # (N, 2) 的数组
            st.success(f"你点击了 {len(points)} 个点！")

        if st.button("生成掩码"):
            predictor.set_image(image_np)
            # input_point = np.array([[x, y]])
            # input_label = np.array([1])  # foreground
            input_point = points  # 多个点
            input_label = np.ones(input_point.shape[0])  # 每个点都是前景(1)

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            mask = masks[0]
            alpha = 0.4
            vis = image_np.copy()
            vis[mask] = (vis[mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8) # 掩码区域着色为绿色
            st.image(vis, caption="掩码结果")

            image_rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
            image_rgba[..., 3] = (mask * 255).astype(np.uint8)  # 掩码区域透明度为255，其它为0
            masked_image = image_rgba
            st.image(masked_image, caption="抠图效果（背景透明）")

            from io import BytesIO
            buffer = BytesIO()
            save_image = Image.fromarray(masked_image)
            save_image.save(buffer, format="PNG")
            buffer.seek(0)
            st.download_button(
                label="下载抠图图片",
                data=buffer,
                file_name="masked_image.png",
                mime="image/png"
            )