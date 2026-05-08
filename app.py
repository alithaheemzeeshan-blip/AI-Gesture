import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Gesture Recognition Game", layout="centered")

st.title("🤖 AI Gesture Recognition Game")
st.write("Upload a hand image and AI will detect brightness-based gestures.")

uploaded_file = st.file_uploader(
    "Upload Hand Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write(f"Image Brightness Score: {brightness:.2f}")

    if brightness > 170:
        gesture = "🖐 Open Palm"
        action = "Start Game"

    elif brightness > 120:
        gesture = "✌ Two Fingers"
        action = "Move Down"

    elif brightness > 80:
        gesture = "☝ One Finger"
        action = "Move Up"

    else:
        gesture = "✊ Fist"
        action = "Pause Game"

    st.success(f"Detected Gesture: {gesture}")
    st.info(f"Game Action: {action}")

else:
    st.warning("Please upload an image.")
