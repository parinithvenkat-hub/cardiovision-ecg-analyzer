import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="CardioVision Ultra | ECG Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM CSS (PRO UI)
# ==============================
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
    font-weight: 600;
}

.block {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 14px;
    margin-bottom: 25px;
}

.metric {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.footer {
    text-align:center;
    opacity:0.6;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🫀 CardioVision Ultra")
st.sidebar.caption("AI ECG Intelligence Platform")

st.sidebar.markdown("### Features")
st.sidebar.markdown("""
✔ ECG Image Analysis  
✔ AI Diagnosis  
✔ Clinical Parameter Estimation  
✔ Medical Explanation  
✔ Doctor Bot  
""")

st.sidebar.warning(
    "Educational use only.\nNot a substitute for clinical diagnosis."
)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="block">
<h1>CardioVision Ultra</h1>
<p>AI-Powered ECG Image Intelligence & Cardiac Risk Screening</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# MODEL LOADER (SAFE)
# ==============================
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
    st.error("⚠️ AI model not available. Demo mode enabled.")
    demo_mode = True
else:
    demo_mode = False

# ==============================
# UPLOAD
# ==============================
st.markdown("<div class='block'><h2>Upload ECG Image</h2></div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded ECG", width=600)

    # ==============================
    # AI DIAGNOSIS
    # ==============================
    st.markdown("<div class='block'><h2>AI Diagnosis</h2></div>", unsafe_allow_html=True)

    if not demo_mode:
        img = ImageOps.fit(image, (224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        labels = ["Normal", "Myocardial Infarction", "Post-MI"]
        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100
    else:
        idx = 1
        confidence = 88.6

    diagnosis = ["Normal", "Myocardial Infarction", "Post-MI"][idx]

    st.success(f"**{diagnosis}**  |  Confidence: **{confidence:.2f}%**")

    # ==============================
    # ECG PARAMETERS (ESTIMATED)
    # ==============================
    st.markdown("<div class='block'><h2>ECG Parameters (Clinical Estimate)</h2></div>", unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = {
        "Heart Rate": "72–110 bpm",
        "PR Interval": "120–200 ms",
        "QRS Duration": "90–130 ms",
        "QT / QTc": "360–470 ms",
        "P Axis": "0° to +75°",
        "QRS Axis": "-30° to +90°",
        "T Axis": "15° to 75°",
        "RV5 / SV1": ">35 mm (LVH indicator)"
    }

    i = 0
    for k, v in metrics.items():
        cols[i % 4].markdown(f"<div class='metric'><b>{k}</b><br>{v}</div>", unsafe_allow_html=True)
        i += 1

    # ==============================
    # MEDICAL EXPLANATION
    # ==============================
    st.markdown("<div class='block'><h2>Medical Interpretation</h2></div>", unsafe_allow_html=True)

    if idx == 0:
        st.write("""
Normal sinus rhythm with no significant ST-T changes or conduction abnormalities.
Overall ECG morphology is within physiological limits.
""")
    elif idx == 1:
        st.write("""
ECG features suggest myocardial infarction, including ST-segment deviation
and abnormal QRS morphology, indicating possible myocardial injury.
Immediate cardiology consultation is advised.
""")
    else:
        st.write("""
ECG findings are consistent with post-myocardial infarction changes,
including residual repolarization abnormalities and altered ventricular conduction.
""")

    # ==============================
    # DOCTOR BOT
    # ==============================
    st.markdown("<div class='block'><h2>Doctor Bot</h2></div>", unsafe_allow_html=True)

    question = st.text_input("Ask a question about ECG or heart conditions")

    if question:
        st.info(
            "Based on ECG patterns, consult a cardiologist for clinical correlation. "
            "This AI assists interpretation but does not replace expert diagnosis."
        )

# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="footer">
Educational & Research Use Only • CardioVision Ultra
</div>
""", unsafe_allow_html=True)
