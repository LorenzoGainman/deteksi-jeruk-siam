import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval

# ===============================
# 1. PAGE CONFIG
# ===============================
st.set_page_config(page_title="Jeruk Siam AI", page_icon="🍊", layout="wide")

# ===============================
# 2. CSS UI
# ===============================
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none; }
video { width: 100% !important; height: auto !important; border-radius: 12px; border: 3px solid #FFA500; }
.instruction-box {
    background-color: #262730;
    color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #FFA500;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 3. DETECT DEVICE
# ===============================
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = screen_width is not None and screen_width < 768

# ===============================
# 4. LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    keras_path = "model_jeruk_rgb_final.keras"

    detector = YOLO("yolov8n.pt")

    if os.path.exists(keras_path):
        classifier = tf.keras.models.load_model(keras_path, compile=False)
    else:
        st.error("Model Keras tidak ditemukan.")
        classifier = None

    return detector, classifier

detector, classifier = load_models()

# ===============================
# 5. UI TITLE
# ===============================
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

# ===============================
# 6. VIDEO PROCESSOR
# ===============================
class OrangeAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.orange_memory = {}
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        current_time = time.time()

        # hapus objek lama (>2 detik hilang)
        to_delete = [
            obj_id for obj_id, data in self.orange_memory.items()
            if current_time - data["last_seen"] > 2.0
        ]
        for obj_id in to_delete:
            del self.orange_memory[obj_id]

        results = self.detector.track(
            img,
            persist=True,
            conf=0.5,
            classes=[47, 49],
            verbose=False,
            imgsz=320,
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id_obj in zip(boxes, ids):
                x1, y1, x2, y2 = box

                if id_obj not in self.orange_memory:
                    self.orange_memory[id_obj] = {
                        "scores": [],
                        "decision": None,
                        "last_seen": current_time,
                    }

                mem = self.orange_memory[id_obj]
                mem["last_seen"] = current_time

                if mem["decision"] is None:
                    crop = img[y1:y2, x1:x2]

                    if crop.size > 0 and self.classifier:
                        input_crop = cv2.resize(crop, (224, 224))
                        input_crop = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB)
                        input_crop = input_crop.astype("float32") / 255.0
                        input_crop = np.expand_dims(input_crop, axis=0)

                        prediction = self.classifier.predict(input_crop, verbose=0)
                        raw_score = float(prediction[0][0])
                        mem["scores"].append(raw_score)

                        temp_label = "MANIS" if raw_score > 0.12 else "ASAM"
                        color = (0, 255, 0) if raw_score > 0.12 else (0, 0, 255)

                        progress = int((len(mem["scores"]) / 15) * 100)
                        label_text = f"{temp_label} ({progress}%)"

                        if len(mem["scores"]) >= 15:
                            avg = sum(mem["scores"]) / len(mem["scores"])
                            mem["decision"] = "MANIS" if avg > 0.12 else "ASAM"
                    else:
                        label_text = "Mencari..."
                        color = (255, 255, 255)
                else:
                    label_text = mem["decision"]
                    color = (0, 255, 0) if label_text == "MANIS" else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===============================
# 7. RTC CONFIG
# ===============================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===============================
# 8. RUN CAMERA
# ===============================
webrtc_streamer(
    key="jeruk-app-final",
    video_processor_factory=OrangeAnalyzer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": {"ideal": 640}},
        "audio": False,
    },
    async_processing=True,
)
