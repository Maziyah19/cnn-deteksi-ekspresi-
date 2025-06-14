import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deteksi Ekspresi Wajah", layout="centered")

st.title("😊 Deteksi Ekspresi Wajah")
st.markdown("Unggah gambar wajah untuk mendeteksi ekspresi menggunakan Deep Learning (CNN).")

uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Mendeteksi ekspresi wajah..."):
        try:
            result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            st.success(f"Ekspresi Terdeteksi: **{emotion.upper()}** 🎯")
        except Exception as e:
            st.error(f"Gagal mendeteksi ekspresi. Pastikan gambar menunjukkan wajah. ({e})")
