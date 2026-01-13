import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="CardioVision Ultra",
    page_icon="🫀",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.big-title {
    font-size: 48px;
    font-weight: 700;
}
.subtitle {
    font-size: 18px;
    opacity: 0.85;
}
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🫀 CardioVision Ultra</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered ECG Image Classification System</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists("ecg_brain.h5"):
        return None
    try:
        return tf.keras.models.load_model("ecg_brain.h5", compile=False)
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file missing or corrupted (`ecg_brain.h5`).")
    st.stop()

# ---------------- UPLOAD ----------------
st.markdown("### 📤 Upload ECG Image")
uploaded = st.file_uploader(
    "PNG / JPG only",
    type=["png", "jpg", "jpeg"]
)

if uploaded is None:
    st.info("⬆️ Please upload an ECG image to start analysis.")
    st.stop()

# ---------------- IMAGE PROCESS ----------------
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded ECG", width=420)

IMG_SIZE = 224  # MUST match training
img = image.resize((IMG_SIZE, IMG_SIZE))
arr = np.array(img).astype("float32") / 255.0
arr = np.expand_dims(arr, axis=0)

# ---------------- PREDICTION ----------------
preds = model.predict(arr, verbose=0)[0]

labels = [
    "Normal",
    "Myocardial Infarction",
    "Post-MI"
]

idx = int(np.argmax(preds))
confidence = float(preds[idx]) * 100

# ---------------- RESULT ----------------
st.markdown("---")
st.markdown("## 🧠 AI Diagnosis")

st.success(f"**{labels[idx]}** — Confidence: **{confidence:.2f}%**")

# ---------------- MEDICAL EXPLANATION ----------------
st.markdown("## 🩺 Medical Interpretation")

if idx == 0:
    st.write("""
• Normal sinus rhythm  
• No pathological ST-segment changes  
• No abnormal Q-waves detected  
""")

elif idx == 1:
    st.write("""
• ST-segment elevation / depression patterns detected  
• Possible pathological Q-waves  
• Suggestive of myocardial tissue injury  
""")

else:
    st.write("""
• Residual ECG abnormalities  
• Scar-related electrical changes  
• Seen after previous myocardial infarction  
""")

# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.warning("⚠️ Educational use only. Not a substitute for clinical diagnosis.")
