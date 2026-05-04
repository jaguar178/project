import streamlit as st
from PIL import Image
from utils.predict import predict_image

st.title("🏍️ Motorrad Erkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild")

    with st.spinner("Analysiere..."):
        try:
            label, confidence = predict_image(image)
        except Exception as e:
            st.error(f"Fehler: {e}")
            st.stop()

    st.success(f"Label: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")
