import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="CardioVision – ECG Analyzer",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 CardioVision: ECG Image Analyzer")
st.write("Upload an ECG image to get AI-based diagnosis")

st.info("🔄 Loading AI model, please wait...")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ecg_brain.h5", compile=False)
    return model

model = load_model()

uploaded = st.file_uploader(
    "Upload ECG image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded ECG", width=400)

    img = ImageOps.fit(image, (224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    labels = ["Normal", "Myocardial Infarction (MI)", "Post-MI"]

    idx = np.argmax(preds)

    st.subheader("🧠 AI Diagnosis")
    st.success(f"{labels[idx]} (Confidence: {preds[idx]*100:.2f}%)")

    st.subheader("🩺 Medical Explanation")
    if idx == 0:
        st.write("Normal sinus rhythm with no critical abnormalities.")
    elif idx == 1:
        st.write("ECG shows patterns associated with myocardial infarction.")
    else:
        st.write("ECG indicates post-MI structural or electrical changes.")

st.markdown("---")
st.caption("⚠️ Educational use only. Not a medical diagnosis.")
