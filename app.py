import streamlit as st
from ultralytics import YOLO, SAM
import torch
import os
import tempfile
import cv2
from PIL import Image
import numpy as np

# Load YOLO model
YOLO_MODEL_PATH = "E:/TumorModel/runs/detect/train/weights/best.pt"
SAM_MODEL_PATH = "E:/TumorModel/sam2_b.pt"

yolo_model = YOLO(YOLO_MODEL_PATH)

# Detect and optionally segment

def detect_and_segment(image, use_sam):
    results = yolo_model(image)
    annotated_image = results[0].plot()
    output_text = ""

    names = yolo_model.names  # Get class names
    detected_classes = []

    if use_sam:
        # Load SAM model only when needed
        try:
            sam_model = SAM(SAM_MODEL_PATH)
        except RuntimeError as e:
            return annotated_image, f"Failed to load SAM model: {e}"

        boxes = results[0].boxes
        class_ids = boxes.cls.int().tolist()

        if len(class_ids):
            xyxy_boxes = boxes.xyxy.cpu()
            orig_img = results[0].orig_img
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam_results = sam_model(orig_img, bboxes=xyxy_boxes, verbose=False, save=False, device=device)
            mask = sam_results[0].masks.data[0].cpu().numpy() * 255
            mask = mask.astype(np.uint8)
            annotated_image = cv2.addWeighted(orig_img, 0.8, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
            detected_classes = [names[c] for c in class_ids]
            class_str = ", ".join(set(detected_classes))
            output_text = f"Detected and segmented tumor types: {class_str}"
        else:
            output_text = "No tumor detected. Skipping SAM."
    else:
        boxes = results[0].boxes
        class_ids = boxes.cls.int().tolist()
        if len(class_ids):
            detected_classes = [names[c] for c in class_ids]
            class_str = ", ".join(set(detected_classes))
            output_text = f"Detected tumor types: {class_str}"
        else:
            output_text = "No tumor detected."

    return annotated_image, output_text

# Streamlit UI
st.set_page_config(page_title="Tumor Detector", layout="centered")
st.title("🧠 Tumor Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
use_sam = st.checkbox("Enable Tumor Segmentation (SAM)", value=True)

if uploaded_file is not None:
    # Convert to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting tumors..."):
            output_img, msg = detect_and_segment(tmp_path, use_sam)
            st.image(output_img, caption=msg, use_column_width=True)

else:
    st.info("Please upload an image to get started.")
