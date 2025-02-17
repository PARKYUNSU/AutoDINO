import streamlit as st

# Streamlit UI 구성
st.set_page_config(layout="wide", page_title="Grounding DINO Auto Labeling")
import os
import torch
import numpy as np
import cv2
import gdown
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate

# 환경 설정
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# HOME 경로 설정
HOME = os.path.expanduser("~")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino", "config", "GroundingDINO_SwinB_cfg.py")
CONFIG_PATH = os.path.join(BASE_DIR, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
# WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "groundingdino_swint_ogc.pth")

# # swinB
# GDRIVE_FILE_ID = "1IhofLclAZhC6j64GpWGjEySWzvk6NHRa"
# GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# swint
GDRIVE_FILE_ID = "1HTxQkiZd3M-p47FggotOhOuFpEY6vHNx"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(WEIGHTS_PATH):
    st.warning("⏳ Downloading model weights... (This may take a few minutes)")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    gdown.download(GDRIVE_URL, WEIGHTS_PATH, quiet=False)

if not os.path.exists(WEIGHTS_PATH):
    st.error("❌ Model weights failed to download!")
else:
    st.success("✅ Model weights successfully downloaded!")

# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO 모델 로드
model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)

st.title("🦖 Grounding DINO Auto Labeling Tool")

# 파일 업로드 기능
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

# 클래스 입력란
class_labels_input = st.sidebar.text_area("✏️ Enter object classes (comma-separated)", "")
class_labels = [c.strip() for c in class_labels_input.split(",") if c.strip()]

# Confidence Threshold 슬라이더 설정
threshold_values = {}
if class_labels:
    for class_name in class_labels:
        threshold_values[class_name] = st.sidebar.slider(
            f"🔍 {class_name} Confidence Threshold",
            min_value=0.1, max_value=0.95, value=0.5, step=0.05
        )

apply_detection = st.sidebar.button("🚀 Apply Detection")

if uploaded_file is not None:
    # 원본 이미지 표시
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    st.image(image_array, caption="📷 Uploaded Image", use_container_width=True)

    # Grounding DINO inference 수행
    image_source, image_tensor = load_image(uploaded_file)

    if apply_detection and class_labels:
        all_boxes = []
        all_logits = []
        all_phrases = []

        for class_name in class_labels:
            text_prompt = class_name
            box_threshold = threshold_values[class_name]
            text_threshold = 0.25  # 고정

            boxes, logits, phrases = predict(
                model=model,
                device=device,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            for i, phrase in enumerate(phrases):
                if phrase.lower() == class_name.lower():
                    all_boxes.append(boxes[i])
                    all_logits.append(logits[i])
                    all_phrases.append(phrase)
        
        if isinstance(all_boxes, list) and len(all_boxes) > 0:
            all_boxes = torch.stack(all_boxes)

        # **탐지된 객체가 있는 경우만 표시**
        if len(all_boxes) > 0:
            annotated_frame = annotate(image_source=image_source, boxes=all_boxes, logits=all_logits, phrases=all_phrases)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 결과 표시
            st.image(annotated_frame, caption="📸 Detected Objects", use_container_width=True)

            # 탐지된 객체 리스트 출력
            st.write("### 📋 Detected Objects")
            for i, box in enumerate(all_boxes):
                label = all_phrases[i]
                confidence = all_logits[i]
                st.write(f"**{label}** - Confidence: {confidence:.2f}")

        else:
            st.warning("❌ No objects detected. Try adjusting the confidence threshold.")

    elif apply_detection and not class_labels:
        st.warning("⚠️ Please enter at least one object class to detect.")

else:
    st.info("📌 Upload an image to start detection.")