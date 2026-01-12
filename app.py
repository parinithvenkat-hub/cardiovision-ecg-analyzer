import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioVision – ECG Analyzer",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 CardioVision: ECG Image Analyzer")
st.write("Upload an ECG image to get AI-based diagnosis and report values")

# ---------------- MODEL LOADING ----------------
loading_msg = st.info("🔄 Loading AI model, please wait...")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ecg_brain.h5", compile=False)
    return model

model = load_model()
loading_msg.empty()   # remove loading message

# ---------------- OCR FUNCTION ----------------
def extract_ecg_values(image):
    text = pytesseract.image_to_string(image)

    values = {}

    patterns = {
        "Heart Rate (HR)": r"(HR|Heart Rate)\s*[:\-]?\s*(\d+)",
        "PR Interval (ms)": r"PR\s*[:\-]?\s*(\d+)",
        "QRS Duration (ms)": r"QRS\s*[:\-]?\s*(\d+)",
        "QT Interval (ms)": r"QT\s*[:\-]?\s*(\d+)",
        "QTc (ms)": r"QTc\s*[:\-]?\s*(\d+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            values[key] = match.group(2)

    return values

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload ECG image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded ECG", width=400)

    # Resize for model
    img = ImageOps.fit(image, (224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    preds = model.predict(arr)[0]
    labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]
    idx = np.argmax(preds)

    # ---------------- RESULTS ----------------
    st.subheader("🧠 AI Diagnosis")
    st.success(f"{labels[idx]} (Confidence: {preds[idx]*100:.2f}%)")

    st.subheader("🩺 Medical Explanation")
    if idx == 0:
        st.write("Normal sinus rhythm with no critical abnormalities detected.")
    elif idx == 1:
        st.write(
            "ECG patterns are suggestive of myocardial infarction. "
            "This may include ST-segment deviations, abnormal Q waves, or T-wave changes."
        )
    else:
        st.write(
            "ECG indicates post-myocardial infarction changes, "
            "which may reflect previous cardiac injury or remodeling."
        )

    # ---------------- OCR VALUES ----------------
    st.subheader("📄 ECG Report Values (OCR)")

    ecg_values = extract_ecg_values(image)

   if ecg_values:
elif not OCR_AVAILABLE:
    st.warning(
        "⚠️ OCR-based ECG value extraction is not supported on cloud hosting. "
        "AI diagnosis is still valid."
        for k, v in ecg_values.items():
            st.write(f"**{k}:** {v}")
    else:
        st.warning("⚠️ No readable ECG values found on this image.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ OCR values are extracted from printed ECG reports, not computed from raw ECG signals.")
st.caption("⚠️ Educational use only. Not a medical diagnosis.")
