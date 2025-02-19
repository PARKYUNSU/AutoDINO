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
    model.eval()  # ✅ 평가 모드 설정 (추론 전환)
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

apply_detection = st.sidebar.button("🚀 Apply Detection")

# 세션 상태에 업로드 파일의 bytes와 detection 결과 캐싱용 딕셔너리 초기화
if "file_bytes" not in st.session_state:
    st.session_state["file_bytes"] = None
if "detection_results" not in st.session_state:
    st.session_state["detection_results"] = {}  # { class_name: (boxes, logits, phrases) }
if "class_thresholds" not in st.session_state:
    st.session_state["class_thresholds"] = {}  # { class_name: threshold }

# 업로드된 파일이 있을 경우: 파일 bytes를 세션에 저장(한번 저장되면 유지)
if uploaded_file is not None:
    # 파일이 새로 업로드되었거나 세션에 저장된 값이 없으면 업데이트
    if st.session_state["file_bytes"] is None:
        st.session_state["file_bytes"] = uploaded_file.read()
    else:
        # 업로드 위젯은 매번 새 파일 객체를 반환하므로, 
        # 파일이 업로드되어 있다면 세션에 저장된 파일을 계속 사용함.
        pass

if st.session_state["file_bytes"] is not None:
    try:
        # 세션에 저장된 파일 bytes를 이용해 원본 이미지를 로드
        original_image = Image.open(io.BytesIO(st.session_state["file_bytes"])).convert("RGB")
        original_array = np.array(original_image)
        
        # detection 실행 전에는 원본 이미지를 보여줌
        if not apply_detection:
            st.image(original_array, caption="📷 Uploaded Image", use_container_width=True)
        
        # 모델 입력용 이미지 로드 (이미지 source, image tensor)
        image_source, image_tensor = load_image(io.BytesIO(st.session_state["file_bytes"]))
        gc.collect()

        all_boxes = []
        all_logits = []
        all_phrases = []

        if apply_detection and class_labels:
            with torch.no_grad():
                for class_name in class_labels:
                    current_threshold = threshold_values[class_name]
                    
                    # 기존에 저장된 결과와 threshold가 동일하면 재사용
                    if (class_name in st.session_state["detection_results"] and
                        st.session_state["class_thresholds"].get(class_name) == current_threshold):
                        boxes, logits, phrases = st.session_state["detection_results"][class_name]
                    else:
                        # 해당 클래스에 대해 새로 detection 실행
                        boxes, logits, phrases = predict(
                            model=model,
                            device=device,
                            image=image_tensor,
                            caption=class_name,
                            box_threshold=current_threshold,
                            text_threshold=0.25  # 고정 값
                        )
                        # 클래스 이름과 일치하는 결과만 필터링
                        filtered_boxes = []
                        filtered_logits = []
                        filtered_phrases = []
                        for i, phrase in enumerate(phrases):
                            if phrase.lower() == class_name.lower():
                                filtered_boxes.append(boxes[i])
                                filtered_logits.append(logits[i])
                                filtered_phrases.append(phrase)
                        if filtered_boxes:
                            filtered_boxes = torch.stack(filtered_boxes)
                        boxes, logits, phrases = filtered_boxes, filtered_logits, filtered_phrases
                        # 결과와 현재 threshold를 세션에 저장
                        st.session_state["detection_results"][class_name] = (boxes, logits, phrases)
                        st.session_state["class_thresholds"][class_name] = current_threshold

                    # 결과가 존재하면 전체 결과에 추가
                    if boxes is not None and len(boxes) > 0:
                        all_boxes.append(boxes)
                        all_logits.extend(logits)
                        all_phrases.extend(phrases)

                # 모든 클래스 결과 합치기
                if all_boxes:
                    all_boxes = torch.cat(all_boxes)
                    
            if all_boxes is not None and len(all_boxes) > 0:
                annotated_frame = annotate(
                    image_source=image_source,
                    boxes=all_boxes,
                    logits=all_logits,
                    phrases=all_phrases
                )
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # detection 결과만 보여줌 (원본 이미지는 더 이상 보이지 않음)
                st.image(annotated_frame, caption="📸 Detected Objects", use_container_width=True)
                
                st.write("### 📋 Detected Objects")
                for i, box in enumerate(all_boxes):
                    label = all_phrases[i]
                    confidence = all_logits[i]
                    st.write(f"**{label}** - Confidence: {confidence:.2f}")

                boxes_list = all_boxes.tolist()
                yolo_lines = yolo_to_txt(boxes_list, all_phrases, class_labels)
                yolo_text = "\n".join(yolo_lines)
                file_name = uploaded_file.name if uploaded_file is not None else "detection_results.txt"
                txt_file_name = f"{os.path.splitext(file_name)[0]}.txt"
                st.download_button(
                    label="Download YOLO Labels",
                    data=yolo_text,
                    file_name=txt_file_name,
                    mime="text/plain")                

            else:
                st.warning("❌ No objects detected. Try adjusting the confidence threshold.")

        elif apply_detection and not class_labels:
            st.warning("⚠️ Please enter at least one object class to detect.")

    finally:
        for var_name in ["image_source", "image_tensor"]:
            if var_name in locals():
                del locals()[var_name]
        gc.collect()
        torch.cuda.empty_cache()

else:
    st.info("📌 Upload an image to start detection.")