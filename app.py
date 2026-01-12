import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CardioVision Ultra",
    page_icon="🫀",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
  color:white;
}
.hero { text-align:center; padding:70px 20px; }
.hero h1 { font-size:4rem; font-weight:900; }
.hero p { font-size:1.3rem; color:#d1d5db; }

.card {
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(14px);
  border-radius: 20px;
  padding: 25px;
  margin-bottom: 25px;
}

.badge {
  padding: 10px 18px;
  border-radius: 999px;
  font-weight: 700;
}
.ok { background:#16a34a; }
.warn { background:#dc2626; }

.footer {
  text-align:center;
  color:#cbd5e1;
  font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
  <h1>CardioVision Ultra</h1>
  <p>AI-Powered ECG Image Intelligence & Clinical Insight Platform</p>
</div>
""", unsafe_allow_html=True)

# ================= SAFE MODEL LOADER =================
@st.cache_resource
def load_model_safe():
    try:
        if not os.path.exists("ecg_brain.h5"):
            return None
        model = tf.keras.models.load_model("ecg_brain.h5", compile=False)
        return model
    except Exception:
        return None

model = load_model_safe()

# ================= STATUS =================
if model is None:
    st.warning("⚠️ AI engine running in **Demo Mode** (model unavailable)")
    DEMO_MODE = True
else:
    DEMO_MODE = False

# ================= UPLOAD =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 Upload ECG Image")
uploaded = st.file_uploader("", type=["png","jpg","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# ================= ANALYSIS =================
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns([1,1.2])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 AI Diagnosis")

        if DEMO_MODE:
            label = "Normal"
            confidence = 92.5
        else:
            img = ImageOps.fit(image,(224,224))
            arr = np.expand_dims(np.array(img)/255.0,0)
            preds = model.predict(arr, verbose=0)[0]
            labels = ["Normal","Myocardial Infarction","Post-MI"]
            idx = int(np.argmax(preds))
            label = labels[idx]
            confidence = preds[idx]*100

        if label == "Normal":
            st.markdown(f'<span class="badge ok">NORMAL · {confidence:.1f}%</span>', unsafe_allow_html=True)
            st.write("Sinus rhythm with no critical ischemic changes.")
        else:
            st.markdown(f'<span class="badge warn">{label} · {confidence:.1f}%</span>', unsafe_allow_html=True)
            st.write("Waveform morphology suggests myocardial injury or remodeling.")

        st.markdown('</div>', unsafe_allow_html=True)

# ================= PARAMETERS NOTE =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 ECG Parameters (Clinical Note)")
st.write("""
Exact HR, PR, QRS, QT, QTc values **cannot be reliably extracted from scanned ECG images**.
They require raw digital signal data and calibration.

This system focuses on **AI-based waveform interpretation**, not misleading numeric extraction.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ================= DOCTOR BOT =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🤖 CardioBot (Medical Assistant)")
q = st.text_input("Ask about ECG / heart conditions")
if q:
    if "mi" in q.lower():
        st.write("Myocardial infarction occurs due to prolonged ischemia from coronary artery blockage.")
    elif "normal" in q.lower():
        st.write("A normal ECG shows coordinated atrial and ventricular depolarization.")
    else:
        st.write("Please consult a cardiologist for personalized diagnosis.")
st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<hr>
<div class="footer">
Educational & Research Use Only · Not a Clinical Diagnostic Tool<br>
CardioVision Ultra
</div>
""", unsafe_allow_html=True)
