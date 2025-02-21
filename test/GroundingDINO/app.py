import streamlit as st

# Streamlit UI 구성
st.set_page_config(layout="wide", page_title="Grounding DINO Auto Labeling")
import os
import torch
import numpy as np
import cv2
import gdown
import gc
import io
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate

def resize_image(image, max_size=(800,800)):
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def yolo_to_txt(boxes, phrases, class_names):
    yolo_data = []
    for idx, box in enumerate(boxes):
        class_name = phrases[idx]
        class_id = class_names.index(class_name) if class_name in class_names else -1
        if class_id != -1:
            x_center, y_center, width, height = map(float, box)
            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_data

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
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

@st.cache_resource
def load_dino_model():
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)
    model.eval()
    return model

model = load_dino_model()

# # Grounding DINO 모델 로드
# model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)

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

# "Apply Detection" 버튼 클릭 여부를 상태로 저장
if "detection_trigger" not in st.session_state:
    st.session_state["detection_trigger"] = False

# Detection 버튼 클릭 시 상태 업데이트
if st.sidebar.button("🚀 Apply Detection", key="apply_detection"):
    st.session_state["detection_trigger"] = True  # 버튼이 눌리면 detection 실행

# 세션 상태 초기화
for key in ["file_bytes", "file_name", "annotated_frame", "all_boxes", "all_logits", "all_phrases", "detection_results", "class_thresholds"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 파일 업로드 시 새 파일인지 확인하고 상태 초기화
if uploaded_file is not None:
    new_bytes = uploaded_file.getvalue()
    if st.session_state["file_bytes"] != new_bytes:
        st.session_state["file_bytes"] = new_bytes
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["annotated_frame"] = None
        st.session_state["all_boxes"] = None
        st.session_state["all_logits"] = None
        st.session_state["all_phrases"] = None
        st.session_state["detection_results"] = {}
        st.session_state["class_thresholds"] = {}

# 결과 표시를 위한 placeholder 생성
image_placeholder = st.empty()

# 객체 검출 및 결과 출력
if st.session_state["file_bytes"] is not None:
    try:
        # 원본 이미지 로드 및 축소
        original_image = Image.open(io.BytesIO(st.session_state["file_bytes"])).convert("RGB")
        resized_image = resize_image(original_image.copy(), max_size=(800, 800))
        original_array = np.array(resized_image)

        # 기존 결과가 있으면 Detection이 적용되기 전까지 유지
        if st.session_state["annotated_frame"] is not None:
            image_placeholder.image(st.session_state["annotated_frame"], caption="📸 Detected Objects", use_container_width=True)
        else:
            image_placeholder.image(original_array, caption="📷 Uploaded Image", use_container_width=True)

        # "Apply Detection" 버튼이 눌렸을 때만 실행
        if st.session_state["detection_trigger"]:
            with st.spinner("Running detection..."):
                image_source, image_tensor = load_image(io.BytesIO(st.session_state["file_bytes"]))
                all_boxes, all_logits, all_phrases = [], [], []

                with torch.no_grad():
                    for class_name in class_labels:
                        current_threshold = threshold_values[class_name]

                        # 기존 detection 결과 재사용
                        if (st.session_state["detection_results"] is not None and 
                            class_name in st.session_state["detection_results"] and 
                            st.session_state["class_thresholds"].get(class_name) == current_threshold):
                            boxes, logits, phrases = st.session_state["detection_results"][class_name]
                        else:
                            boxes, logits, phrases = predict(
                                model=model,
                                device=device,
                                image=image_tensor,
                                caption=class_name,
                                box_threshold=current_threshold,
                                text_threshold=0.25
                            )
                            filtered_boxes, filtered_logits, filtered_phrases = [], [], []
                            for i, phrase in enumerate(phrases):
                                if phrase.lower() == class_name.lower():
                                    filtered_boxes.append(boxes[i])
                                    filtered_logits.append(logits[i])
                                    filtered_phrases.append(phrase)
                            if filtered_boxes:
                                filtered_boxes = torch.stack(filtered_boxes)
                            boxes, logits, phrases = filtered_boxes, filtered_logits, filtered_phrases
                            st.session_state["detection_results"][class_name] = (boxes, logits, phrases)
                            st.session_state["class_thresholds"][class_name] = current_threshold

                        if boxes is not None and len(boxes) > 0:
                            all_boxes.append(boxes)
                            all_logits.extend(logits)
                            all_phrases.extend(phrases)

                if all_boxes:
                    all_boxes = torch.cat(all_boxes)

                # 결과 저장
                if all_boxes is not None and len(all_boxes) > 0:
                    annotated_frame = annotate(image_source, all_boxes, all_logits, all_phrases)
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.session_state["annotated_frame"] = annotated_frame
                    st.session_state["all_boxes"] = all_boxes.cpu().numpy()
                    st.session_state["all_logits"] = [float(x) for x in all_logits]
                    st.session_state["all_phrases"] = all_phrases
            st.session_state["detection_trigger"] = False
            del image_tensor
            gc.collect()

        # Detection 결과 표시
        if st.session_state["annotated_frame"] is not None:
            image_placeholder.image(st.session_state["annotated_frame"], caption="📸 Detected Objects", use_container_width=True)
            st.write("### 📋 Detected Objects")
            for i, box in enumerate(st.session_state["all_boxes"].tolist()):
                label = st.session_state["all_phrases"][i]
                confidence = st.session_state["all_logits"][i]
                st.write(f"**{label}** - Confidence: {confidence:.2f}")
            # YOLO 라벨 변환 및 다운로드 버튼
            if st.session_state["all_boxes"] is not None and st.session_state["all_phrases"] is not None:
                boxes_list = st.session_state["all_boxes"].tolist()
                yolo_lines = yolo_to_txt(boxes_list, st.session_state["all_phrases"], class_labels)
                yolo_text = "\n".join(yolo_lines)
                file_name = st.session_state["file_name"] if st.session_state["file_name"] is not None else "detection_results.txt"
                txt_file_name = f"{os.path.splitext(file_name)[0]}.txt"
                st.download_button(
                    label="Download YOLO Labels",
                    data=yolo_text,
                    file_name=txt_file_name,
                    mime="text/plain"
                )
        else:
            st.warning("❌ No objects detected. Try adjusting the confidence threshold.")
    finally:
        try:
            del image_source, image_tensor
        except NameError:
            pass
        gc.collect()
else:
    st.info("📌 Upload an image to start detection.")