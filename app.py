import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Gesture Recognition Game", layout="centered")

st.title("🤖 AI Gesture Recognition Game")
st.write("Upload a hand image and AI will detect your gesture.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x)
    else:
        fingers.append(hand_landmarks.landmark[tips[0]].x > hand_landmarks.landmark[tips[0] - 1].x)

    # Other fingers
    for tip in tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers.count(True), fingers


def detect_gesture(finger_count, fingers):
    thumb, index, middle, ring, pinky = fingers

    if finger_count == 0:
        return "✊ Fist", "Pause Game"

    elif finger_count == 5:
        return "🖐 Open Palm", "Start Game"

    elif index and not middle and not ring and not pinky:
        return "☝ One Finger", "Move Up"

    elif index and middle and not ring and not pinky:
        return "✌ Two Fingers", "Move Down"

    elif thumb and not index and not middle and not ring and not pinky:
        return "👍 Thumbs Up", "Score Point"

    else:
        return "Unknown Gesture", "No Action"


uploaded_file = st.file_uploader("Upload hand image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label

                finger_count, fingers = count_fingers(hand_landmarks, hand_label)
                gesture, action = detect_gesture(finger_count, fingers)

                mp_drawing.draw_landmarks(
                    image_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                st.success("Hand detected successfully!")
                st.subheader(f"Detected Gesture: {gesture}")
                st.subheader(f"Game Action: {action}")
                st.write(f"Fingers Detected: {finger_count}")

        else:
            st.error("No hand detected. Please upload a clear hand image.")

    final_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    st.image(final_image, caption="Processed Image", use_container_width=True)

else:
    st.info("Please upload a hand image to start gesture recognition.")