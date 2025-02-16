import os
import torch
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate

# 환경 설정
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# HOME 경로 설정
HOME = os.path.expanduser("~")

# 모델 및 가중치 경로 설정
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")

# 디바이스 설정 (MPS보다 CPU가 빠를 가능성 있음)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Grounding DINO 모델 로드
model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)

# Streamlit UI 구성
st.set_page_config(layout="wide", page_title="Grounding DINO Auto Labeling")

st.title("🦖 Grounding DINO Auto Labeling Tool")

# 파일 업로드 기능
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 기본 클래스 설정
default_classes = ["sofa", "lamp", "bed", "dresser"]
class_labels = st.sidebar.text_area("Edit All Classes (comma-separated)", ", ".join(default_classes))
class_labels = [c.strip() for c in class_labels.split(",")]

# Confidence Threshold 슬라이더 설정
threshold_values = {}
for class_name in class_labels:
    threshold_values[class_name] = st.sidebar.slider(
        f"{class_name} Confidence Threshold",
        min_value=0.1, max_value=0.95, value=0.5, step=0.05
    )

if uploaded_file is not None:
    # 이미지 로드
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    
    # Grounding DINO inference 수행
    image_source, image_tensor = load_image(uploaded_file)

    # 사용자가 입력한 클래스 기반 탐색 수행
    text_prompt = ", ".join(class_labels)
    box_threshold = min(threshold_values.values())  # 최소 threshold 적용
    text_threshold = 0.25  # 고정

    # Object Detection 수행
    boxes, logits, phrases = predict(
        model=model,
        device=device,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # 결과 Annotate
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # 결과 표시
    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)

    # 탐지된 객체 리스트 출력
    st.write("### Detected Objects")
    for i, box in enumerate(boxes):
        label = phrases[i]
        confidence = logits[i]
        if confidence >= threshold_values.get(label, 0.5):  # 각 클래스 threshold 적용
            st.write(f"**{label}** - Confidence: {confidence:.2f}")

else:
    st.info("Please upload an image to start detection.")