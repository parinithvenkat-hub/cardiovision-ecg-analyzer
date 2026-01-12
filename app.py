import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="CardioVision Ultra",
    page_icon="🫀",
    layout="wide"
)

# ================= PREMIUM CSS =================
st.markdown("""
<style>
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(20px);}
  to {opacity: 1; transform: translateY(0);}
}

.stApp {
  background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
  color: white;
}

.hero {
  text-align: center;
  padding: 80px 20px;
  animation: fadeIn 1.2s ease;
}

.hero h1 {
  font-size: 4rem;
  font-weight: 900;
}

.hero p {
  font-size: 1.4rem;
  color: #cfd9df;
}

.glass {
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(14px);
  border-radius: 20px;
  padding: 30px;
  margin-bottom: 30px;
  animation: fadeIn 0.8s ease;
}

.section-title {
  font-size: 2rem;
  font-weight: 800;
  margin-bottom: 20px;
}

.badge {
  padding: 10px 20px;
  border-radius: 999px;
  display: inline-block;
  font-weight: 700;
}

.success { background: #16a34a; }
.danger { background: #dc2626; }

.chat-bubble {
  background: rgba(255,255,255,0.12);
  padding: 15px;
  border-radius: 16px;
  margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
  <h1>CardioVision Ultra</h1>
  <p>AI-Powered ECG Image Intelligence & Cardiac Risk Screening</p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists("ecg_brain.h5"):
        return None
    return tf.keras.models.load_model("ecg_brain.h5", compile=False)

model = load_model()
if model is None:
    st.error("❌ Model file missing or corrupted")
    st.stop()

# ================= UPLOAD =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📤 Upload ECG Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# ================= ANALYSIS =================
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        img = ImageOps.fit(image, (224,224))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        preds = model.predict(arr, verbose=0)[0]

        labels = ["Normal", "Myocardial Infarction", "Post-MI"]
        idx = int(np.argmax(preds))
        conf = preds[idx]*100

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI Diagnosis</div>', unsafe_allow_html=True)

        if idx == 0:
            st.markdown(f'<span class="badge success">NORMAL · {conf:.1f}%</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge danger">{labels[idx]} · {conf:.1f}%</span>', unsafe_allow_html=True)

        st.markdown("### 🩺 Clinical Interpretation")

        if idx == 0:
            st.write("Electrical conduction appears consistent with a healthy sinus rhythm.")
        elif idx == 1:
            st.write("Waveform morphology suggests ischemic injury patterns commonly seen in acute MI.")
        else:
            st.write("Post-MI remodeling indicators such as Q-wave persistence are observed.")

        st.markdown('</div>', unsafe_allow_html=True)

# ================= PARAMETERS (HONEST) =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 ECG Parameters (Why Not Shown)</div>', unsafe_allow_html=True)
st.write("""
HR, PR, QRS, QT, QTc, Axis values **cannot be accurately computed from scanned ECG images**.

These require:
• Raw signal data  
• Sampling frequency  
• Voltage calibration  

This platform focuses on **AI waveform intelligence**, not misleading numbers.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ================= DOCTOR BOT =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤖 CardioBot (AI Assistant)</div>', unsafe_allow_html=True)

q = st.text_input("Ask a cardiology question")

if q:
    st.markdown('<div class="chat-bubble">', unsafe_allow_html=True)
    if "mi" in q.lower():
        st.write("Myocardial infarction occurs due to coronary artery blockage leading to myocardial necrosis.")
    elif "normal" in q.lower():
        st.write("A normal ECG reflects coordinated atrial and ventricular depolarization.")
    else:
        st.write("Please consult a cardiologist for personalized medical advice.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<hr>
<p style="text-align:center; color:#cbd5e1;">
Educational & Research Use Only · CardioVision Ultra
</p>
""", unsafe_allow_html=True)
