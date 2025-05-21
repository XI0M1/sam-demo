import streamlit as st
from PIL import Image
import numpy as np
from sam_clip_matcher import SAMCLIPMatcher
import os

# 初始化匹配器（只初始化一次）
@st.cache_resource
def load_matcher():
    sam_ckpt = os.path.join("checkpoints", "sam_vit_b_01ec64.pth")
    return SAMCLIPMatcher(sam_checkpoint=sam_ckpt)

st.set_page_config(page_title="SAM + CLIP 匹配掩码", layout="wide")
st.title("🎯 基于文本的图像掩码提取（SAM + CLIP）")

matcher = load_matcher()

# 文件上传
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

# 文本输入
prompt = st.text_input("输入一段英文描述", value="")

if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上传的图片", use_container_width = True)

    with st.spinner("处理中..."):
        vis_result, mask, err = matcher.process(image, prompt)

    if err:
        st.error(err)
    else:
        st.image(vis_result, caption="匹配结果掩码", use_container_width=True)
else:
    st.info("请上传图片并输入提示词。")
