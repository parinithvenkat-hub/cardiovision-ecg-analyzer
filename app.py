import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="CardioVision Ultra | ECG Intelligence",
    page_icon="🫀",
    layout="wide"
)

# ===================== CUSTOM CSS (UI + ANIMATION) =====================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(120deg, #f8fbff, #eef4ff);
}

/* Main Title */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: #0b1f3a;
    animation: fadeIn 1.2s ease-in-out;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #4b5d73;
    margin-bottom: 2rem;
    animation: fadeIn 1.5s ease-in-out;
}

/* Section Headers */
.section-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0b1f3a;
    margin-top: 1.5rem;
}

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    animation: slideUp 0.8s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes slideUp {
    from {transform: translateY(30px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.markdown("## 🫀 CardioVision Ultra")
st.sidebar.markdown("**AI ECG Intelligence Platform**")
st.sidebar.markdown("---")
st.sidebar.markdown("### Capabilities")
st.sidebar.markdown("✔ ECG Image Upload")
st.sidebar.markdown("✔ AI Waveform Diagnosis")
st.sidebar.markdown("✔ Medical Interpretation")
st.sidebar.markdown("✔ Doctor Assistant Bot")
st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Educational Use Only\n\n"
    "Not a substitute for professional medical diagnosis."
)

# ===================== HEADER =====================
st.markdown("<div class='main-title'>CardioVision Ultra</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered ECG Image Analysis & Cardiac Risk Assessment</div>",
    unsafe_allow_html=True
)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ecg_brain.h5", compile=False)

with st.spinner("🧠 Initializing Cardiac AI Engine..."):
    model = load_model()

# ===================== UPLOAD SECTION =====================
st.markdown("<div class='section-title'>📤 Upload ECG Image</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ===================== ANALYSIS =====================
if uploaded:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("<div class='section-title'>🖼 ECG Preview</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>🧠 AI Waveform Diagnosis</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        img = ImageOps.fit(image, (224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]
        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100

        if idx == 0:
            st.success("✅ **Normal ECG Detected**")
        else:
            st.error(f"⚠️ **{labels[idx]} Detected**")

        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.markdown("### 🩺 Medical Interpretation")

        if idx == 0:
            st.write(
                "The ECG waveform reflects a **normal sinus rhythm** "
                "with no significant pathological abnormalities."
            )
        elif idx == 1:
            st.write(
                "The ECG image demonstrates waveform patterns suggestive of "
                "**myocardial infarction**, such as abnormal Q waves or "
                "ST-segment deviations."
            )
        else:
            st.write(
                "The ECG indicates **post-myocardial infarction changes**, "
                "which may reflect prior cardiac injury or remodeling."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ===================== ECG PARAMETERS NOTE =====================
    st.markdown("<div class='section-title'>📊 ECG Parameters Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.warning(
        "Numerical ECG parameters such as **Heart Rate (HR), PR Interval, "
        "QRS Duration, QT/QTc, Axis, RV5/SV1** are computed by ECG machines "
        "using **raw electrical signal data**.\n\n"
        "Scanned ECG images do not contain sufficient timing and voltage "
        "information for accurate calculation.\n\n"
        "**This system therefore focuses on AI-based waveform classification.**"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== DOCTOR BOT =====================
st.markdown("<div class='section-title'>🤖 Doctor Bot – Ask About ECG</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

user_q = st.text_input("Ask a question (e.g., What is myocardial infarction?)")

if user_q:
    if "mi" in user_q.lower() or "myocardial" in user_q.lower():
        st.write(
            "**Doctor Bot:** Myocardial infarction (heart attack) occurs when "
            "blood flow to the heart muscle is blocked, causing tissue damage."
        )
    elif "normal" in user_q.lower():
        st.write(
            "**Doctor Bot:** A normal ECG indicates proper electrical activity "
            "and coordinated heart contractions."
        )
    elif "post" in user_q.lower():
        st.write(
            "**Doctor Bot:** Post-MI ECG changes reflect healing or scarring "
            "after a previous heart attack."
        )
    else:
        st.write(
            "**Doctor Bot:** Please consult a cardiologist for personalized advice. "
            "This tool provides educational insights only."
        )

st.markdown("</div>", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4b5d73;'>"
    "⚠️ For educational & research purposes only | "
    "CardioVision Ultra – AI ECG Intelligence Platform"
    "</p>",
    unsafe_allow_html=True
)
