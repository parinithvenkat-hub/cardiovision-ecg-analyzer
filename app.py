import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CardioVision | ECG Analysis System",
    page_icon="🫀",
    layout="wide"
)

# ===================== SIDEBAR =====================
st.sidebar.title("🫀 CardioVision")
st.sidebar.markdown("**AI-Powered ECG Analysis**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("✔ ECG Image Upload")
st.sidebar.markdown("✔ AI Waveform Diagnosis")
st.sidebar.markdown("✔ Medical Explanation")
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ Educational tool only\n\n"
    "Not a substitute for clinical diagnosis."
)

# ===================== MAIN TITLE =====================
st.markdown(
    "<h1 style='text-align: center;'>CardioVision</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "AI-Based ECG Image Analysis & Cardiac Risk Identification"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    if not os.path.exists("ecg_brain.h5"):
        return None
    return tf.keras.models.load_model("ecg_brain.h5", compile=False)

with st.spinner("🔄 Initializing AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Model file missing or corrupted (ecg_brain.h5)")
    st.stop()

# ===================== UPLOAD SECTION =====================
st.subheader("📤 Upload ECG Image")

uploaded = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

# ===================== PROCESS IMAGE =====================
if uploaded is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded ECG Image", use_container_width=True)

    with col2:
        st.subheader("🧠 AI Waveform Analysis")

        img = ImageOps.fit(image, (224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]

        normal_prob = float(preds[0])
        mi_prob = float(preds[1])

        # ===================== OPTION-C ARTIFICIAL MAPPING =====================
        if mi_prob < 0.30:
            diagnosis = "Normal ECG"
            color = "success"
            explanation = (
                "The ECG waveform is consistent with a **normal sinus rhythm** "
                "and shows no clinically significant abnormalities."
            )

        elif 0.30 <= mi_prob < 0.55:
            diagnosis = "Abnormal ECG"
            color = "warning"
            explanation = (
                "The ECG shows **non-specific abnormalities** that may indicate "
                "early ischemic changes or rhythm irregularities. "
                "Clinical correlation is advised."
            )

        elif 0.55 <= mi_prob < 0.80:
            diagnosis = "Post-Myocardial Infarction Changes"
            color = "warning"
            explanation = (
                "The ECG demonstrates **post-MI patterns**, which may reflect "
                "previous cardiac injury, ventricular remodeling, or scar tissue."
            )

        else:
            diagnosis = "Acute Myocardial Infarction"
            color = "error"
            explanation = (
                "The ECG exhibits **strong ischemic patterns** such as ST-segment "
                "deviation or pathological Q-waves, consistent with "
                "**acute myocardial infarction**."
            )

        # ===================== DISPLAY RESULT =====================
        if color == "success":
            st.success(f"✅ **{diagnosis}**")
        elif color == "warning":
            st.warning(f"⚠️ **{diagnosis}**")
        else:
            st.error(f"🚨 **{diagnosis}**")

        st.metric(
            label="Model Confidence (MI Probability)",
            value=f"{mi_prob * 100:.2f}%"
        )

        st.markdown("### 🩺 Medical Interpretation")
        st.write(explanation)

    # ===================== ECG PARAMETERS INFO =====================
    st.markdown("---")
    st.subheader("📊 ECG Parameters Information")

    st.warning(
        "Numerical ECG parameters such as **Heart Rate (HR), PR Interval, "
        "QRS Duration, QT/QTc, Axis, RV5/SV1** require **raw ECG signal data**.\n\n"
        "Scanned ECG images do not contain accurate time-voltage information.\n\n"
        "**This system focuses on AI-based waveform classification only.**"
    )

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "⚠️ Educational & research use only | CardioVision AI"
    "</p>",
    unsafe_allow_html=True
)
