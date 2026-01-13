import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

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
    return tf.keras.models.load_model("ecg_brain.h5", compile=False)

with st.spinner("🔄 Initializing AI model..."):
    model = load_model()

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
        labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]
        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100

        if idx == 0:
            st.success(f"✅ **Normal ECG**")
        else:
            st.error(f"⚠️ **{labels[idx]} Detected**")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        st.markdown("### 🩺 Medical Interpretation")

        if idx == 0:
            st.write(
                "The ECG waveform demonstrates a **normal sinus rhythm** "
                "with no clinically significant abnormalities."
            )
        elif idx == 1:
            st.write(
                "The ECG image exhibits waveform patterns consistent with "
                "**myocardial infarction**, such as abnormal Q waves or "
                "ST-segment deviations."
            )
        else:
            st.write(
                "The ECG indicates **post-myocardial infarction changes**, "
                "which may reflect previous cardiac injury or structural remodeling."
            )

    # ===================== ECG PARAMETERS INFO =====================
    st.markdown("---")
    st.subheader("📊 ECG Parameters Information")

    st.warning(
        "Numerical ECG parameters such as **Heart Rate (HR), PR Interval, "
        "QRS Duration, QT/QTc, Axis, RV5/SV1** are calculated by ECG machines "
        "using **raw electrical signal data**.\n\n"
        "Scanned ECG images do not contain accurate timing and voltage "
        "information to compute these parameters reliably.\n\n"
        "**Therefore, this system focuses on AI-based waveform classification.**"
    )

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    "⚠️ For educational and research purposes only | "
    "Developed using Deep Learning & Computer Vision"
    "</p>",
    unsafe_allow_html=True
)
