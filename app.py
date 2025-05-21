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
from io import BytesIO

# 必须最前设置页面信息
st.set_page_config(page_title="SAM1 基础交互式分割 🎯", page_icon="📌", layout="centered")

# 标题
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>SAM 本地分割 Demo</h1>",
    unsafe_allow_html=True
)
st.markdown("请上传一张图片进行分割：")

# 上传图片
uploaded_file = st.file_uploader("上传一张图片", type=["png", "jpg", "jpeg"],label_visibility="collapsed")

# 加载SAM模型
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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # 缩放处理
    max_display_width = 800
    h, w = image_np.shape[:2]
    scale = 1.0
    if w > max_display_width:
        scale = max_display_width / w
        display_width = int(w * scale)
        display_height = int(h * scale)
    else:
        display_width = w
        display_height = h

    display_image = cv2.resize(image_np, (display_width, display_height), interpolation=cv2.INTER_AREA)
    display_pil_image = Image.fromarray(display_image)

    # 绘制点击点
    st.subheader("点击图像选择前景点")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=10,
        background_image=display_pil_image,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        points = []
        for obj in canvas_result.json_data["objects"]:
            x = int(obj["left"] / scale)  # 缩放回原图
            y = int(obj["top"] / scale)
            points.append([x, y])
        points = np.array(points)

        st.success(f"✅ 你点击了 {len(points)} 个点！")

        if st.button("🎯 生成掩码"):
            predictor.set_image(image_np)
            input_point = points
            input_label = np.ones(input_point.shape[0])  # 全部作为前景

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            mask = masks[0]

            # 可视化叠加
            alpha = 0.4
            vis = image_np.copy()
            vis[mask] = (vis[mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
            st.image(vis, caption="半透明掩码叠加效果", use_container_width=True)

            # 抠图
            image_rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
            image_rgba[..., 3] = (mask * 255).astype(np.uint8)
            masked_image = image_rgba

            st.image(masked_image, caption="抠图（背景透明）", use_container_width=True)

            # 下载按钮
            buffer = BytesIO()
            save_image = Image.fromarray(masked_image)
            save_image.save(buffer, format="PNG")
            buffer.seek(0)
            st.markdown("<h3 style='text-align: center;'>下载你的抠图图片</h3>", unsafe_allow_html=True)
            st.download_button(
                label="下载抠图",
                data=buffer,
                file_name="masked_image.png",
                mime="image/png",
                key="download-button",
            )
