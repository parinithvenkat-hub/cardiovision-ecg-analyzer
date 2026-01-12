import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioVision – ECG Analyzer",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 CardioVision: ECG Image Analyzer")
st.write("Upload an ECG image to get AI-based waveform diagnosis")

# ---------------- MODEL LOADING ----------------
loading_msg = st.info("🔄 Loading AI model, please wait...")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ecg_brain.h5", compile=False)
    return model

model = load_model()
loading_msg.empty()

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload ECG image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded ECG", width=400)

    # Resize image to model input size
    img = ImageOps.fit(image, (224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    preds = model.predict(arr)[0]
    labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100

    # ---------------- RESULTS ----------------
    st.subheader("🧠 AI Waveform Diagnosis")

    if idx == 0:
        st.success(f"Normal ECG ({confidence:.2f}% confidence)")
    else:
        st.error(f"{labels[idx]} ({confidence:.2f}% confidence)")

    st.subheader("🩺 Medical Explanation")

    if idx == 0:
        st.write(
            "The ECG waveform shows a normal sinus rhythm with regular intervals "
            "and no significant abnormalities."
        )
    elif idx == 1:
        st.write(
            "The ECG waveform indicates patterns consistent with myocardial infarction, "
            "such as ST-segment deviations or abnormal Q waves."
        )
    else:
        st.write(
            "The ECG waveform suggests post-myocardial infarction changes, "
            "which may reflect previous cardiac injury or structural remodeling."
        )

    # ---------------- ECG VALUES SECTION ----------------
    st.subheader("📄 ECG Parameters (HR, PR, QRS, QT, QTc)")

    st.warning(
        "⚠️ ECG numerical values such as Heart Rate (HR), PR interval, "
        "QRS duration, QT/QTc, Axis, and RV5/SV1 are calculated by ECG machines "
        "from raw signal data.\n\n"
        "Scanned ECG images do not contain sufficient timing and voltage "
        "information to compute these values accurately.\n\n"
        "In this cloud-deployed version, the system focuses on AI-based "
        "waveform classification. OCR-based report reading is supported "
        "only in the local version."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠️ Educational use only. Not a substitute for professional medical diagnosis.")
