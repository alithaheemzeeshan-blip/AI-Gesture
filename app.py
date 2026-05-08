import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="AI Live Gesture Recognition", layout="centered")

st.title("🤖 AI Live Hand Gesture Recognition Game")
st.write("Allow camera permission and show your hand gesture.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
    else:
        fingers.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)

    for tip in tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    return fingers.count(True), fingers


def detect_gesture(finger_count, fingers):
    thumb, index, middle, ring, pinky = fingers

    if finger_count == 0:
        return "Fist", "Pause Game"

    elif finger_count == 5:
        return "Open Palm", "Start Game"

    elif index and not middle and not ring and not pinky:
        return "One Finger", "Move Up"

    elif index and middle and not ring and not pinky:
        return "Two Fingers", "Move Down"

    elif thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up", "Score Point"

    else:
        return "Unknown", "No Action"


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture_text = "No Hand Detected"
        action_text = "Show your hand"

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                finger_count, fingers = count_fingers(hand_landmarks, hand_label)
                gesture, action = detect_gesture(finger_count, fingers)

                gesture_text = f"Gesture: {gesture}"
                action_text = f"Action: {action}"

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(img, gesture_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.putText(img, action_text, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="gesture-live-camera",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
