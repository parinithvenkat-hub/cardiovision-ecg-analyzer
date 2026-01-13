import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CardioVision Ultra | ECG Intelligence",
    page_icon="🫀",
    layout="wide"
)

# ================= BASIC STYLING =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #f7faff, #eef3fb);
}
.title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: #0b1f3a;
}
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #4b5d73;
    margin-bottom: 2rem;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.section {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0b1f3a;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='title'>CardioVision Ultra</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered ECG Image Analysis & Cardiac Risk Screening</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.title("🫀 CardioVision Ultra")
st.sidebar.markdown("AI ECG Analysis System")
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("✔ ECG Image Upload")
st.sidebar.markdown("✔ AI Diagnosis")
st.sidebar.markdown("✔ Medical Explanation")
st.sidebar.markdown("✔ Doctor Bot")
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Educational Use Only\n\n"
    "Not a substitute for clinical diagnosis."
)

# ================= MODEL LOADING =================
@st.cache_resource
def load_model_safe():
    if not os.path.exists("ecg_brain.h5"):
        return None
    try:
        model = tf.keras.models.load_model(
            "ecg_brain.h5",
            compile=False
        )
        return model
    except Exception:
        return None

with st.spinner("🧠 Initializing AI engine..."):
    model = load_model_safe()

if model is None:
    st.error(
        "❌ AI model could not be loaded.\n\n"
        "Please ensure **ecg_brain.h5** is present in the GitHub repository root "
        "and is not corrupted."
    )
    st.stop()

# ================= UPLOAD =================
st.markdown("<div class='section'>📤 Upload ECG Image</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYSIS =================
if uploaded is not None:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("<div class='section'>🖼 ECG Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section'>🧠 AI Diagnosis</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        img = ImageOps.fit(image, (224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100

        if idx == 0:
            st.success("✅ **Normal ECG Detected**")
        else:
            st.error(f"⚠️ **{labels[idx]} Detected**")

        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.markdown("### 🩺 Medical Interpretation")

        if idx == 0:
            st.write(
                "The ECG waveform indicates a **normal sinus rhythm** "
                "with no clinically significant abnormalities."
            )
        elif idx == 1:
            st.write(
                "The ECG image shows patterns suggestive of "
                "**myocardial infarction**, such as abnormal Q waves "
                "or ST-segment deviations."
            )
        else:
            st.write(
                "The ECG indicates **post-myocardial infarction changes**, "
                "likely due to previous cardiac injury or remodeling."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ================= ECG PARAMETERS NOTE =================
    st.markdown("<div class='section'>📊 ECG Parameters Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.warning(
        "Numerical ECG parameters such as **Heart Rate (HR), PR Interval, "
        "QRS Duration, QT/QTc, Axis, RV5/SV1** are calculated by ECG machines "
        "using **raw electrical signal data**.\n\n"
        "Scanned ECG images do not contain sufficient timing and voltage "
        "information for accurate computation.\n\n"
        "**This system therefore focuses on AI-based waveform classification.**"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ================= DOCTOR BOT =================
st.markdown("<div class='section'>🤖 Doctor Bot</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

question = st.text_input("Ask a question about ECG or heart conditions")

if question:
    q = question.lower()
    if "mi" in q or "heart attack" in q:
        st.write(
            "**Doctor Bot:** Myocardial infarction occurs when blood flow "
            "to part of the heart muscle is blocked, causing tissue damage."
        )
    elif "normal" in q:
        st.write(
            "**Doctor Bot:** A normal ECG indicates proper electrical "
            "activity and coordinated heart contractions."
        )
    elif "post" in q:
        st.write(
            "**Doctor Bot:** Post-MI ECG changes reflect healing or scarring "
            "after a previous heart attack."
        )
    else:
        st.write(
            "**Doctor Bot:** This system provides educational information. "
            "For medical advice, consult a cardiologist."
        )

st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4b5d73;'>"
    "⚠️ Educational & Research Use Only | CardioVision Ultra"
    "</p>",
    unsafe_allow_html=True
)
