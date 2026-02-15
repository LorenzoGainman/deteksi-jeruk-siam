import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from streamlit_js_eval import streamlit_js_eval

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Jeruk Siam AI", page_icon="🍊", layout="wide")

# CSS: Tampilan Antarmuka (Sudah ditambahkan CSS Adaptif Mobile)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none; }
    .element-container iframe { width: 100% !important; }
    video { width: 100% !important; height: auto !important; border-radius: 12px; border: 3px solid #FFA500; }
    .instruction-box {
        background-color: #262730;
        color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #FFA500;
        margin-bottom: 20px;
        font-size: 16px;
        line-height: 1.6;
    }
    .instruction-box h4 { color: #FFA500; margin-top: 0; margin-bottom: 10px; }
    .tech-tag {
        font-size: 12px;
        color: #aaaaaa;
        margin-top: 15px;
        font-style: italic;
        border-top: 1px solid #444;
        padding-top: 8px;
    }
    
    /* Tambahan CSS Adaptif untuk Mobile */
    @media (max-width: 768px) {
        .element-container iframe { 
            height: 500px !important; 
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. DETEKSI DEVICE ---
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = screen_width is not None and screen_width < 768

# --- 3. LOAD & PREPARE MODELS ---
@st.cache_resource
def load_and_prepare_models():
    keras_path = 'model_jeruk_rgb_final.keras'
    tflite_path = 'model_jeruk_siam.tflite'

    if not os.path.exists(tflite_path) and os.path.exists(keras_path):
        try:
            model = tf.keras.models.load_model(keras_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            del model
        except Exception as e:
            st.error(f"Gagal konversi model: {e}")

    detector = YOLO('yolov8n.pt')

    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        classifier = interpreter
        model_type = "tflite"
    else:
        classifier = tf.keras.models.load_model(keras_path)
        model_type = "keras"

    return detector, classifier, model_type

detector, classifier, model_type = load_and_prepare_models()

# --- 4. UI: PANDUAN PENGGUNA ---
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

if is_mobile:
    st.markdown("""
    <div class="instruction-box">
        <h4>📱 Panduan Penggunaan Mobile:</h4>
        <ol>
            <li>Letakkan <b>Buah Jeruk</b> pada posisi tetap dengan cahaya terang.</li>
            <li>Tekan <b>START</b> dan arahkan kamera ke buah.</li>
            <li>Jaga jarak stabil <b>15 - 30 cm</b>.</li>
            <li>Tahan posisi selama <b>5 detik</b> hingga analisa selesai 100%.</li>
        </ol>
        <div class="tech-tag">Engine: YOLOv8 Tracking & MobileNetV2 Classification</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="instruction-box">
        <h4>💻 Panduan Penggunaan PC/Laptop:</h4>
        <ol>
            <li>Tekan tombol <b>START</b> untuk mengaktifkan webcam.</li>
            <li>Pegang buah jeruk dan dekatkan ke arah kamera (jarak 15-30 cm).</li>
            <li>Pastikan pencahayaan cukup terang agar warna kulit terlihat jelas.</li>
            <li>Tahan posisi hingga label <b>MANIS</b> atau <b>ASAM</b> terkunci.</li>
        </ol>
        <div class="tech-tag">Engine: YOLOv8 Tracking & MobileNetV2 Classification</div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. ENGINE PEMROSESAN VIDEO ---
class OrangeAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.model_type = model_type
        self.orange_memory = {}
        self.frame_count = 0 
        
        if model_type == "tflite":
            self.input_details = classifier.get_input_details()
            self.output_details = classifier.get_output_details()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % 3 != 0:
            return img 

        current_time = time.time()
        
        to_delete = [obj_id for obj_id, data in self.orange_memory.items() 
                     if current_time - data["last_seen"] > 2.0]
        for obj_id in to_delete:
            del self.orange_memory[obj_id]

        results = self.detector.track(img, persist=True, conf=0.5, classes=[47, 49], verbose=False, imgsz=320)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id_obj in zip(boxes, ids):
                x1, y1, x2, y2 = box
                if id_obj not in self.orange_memory:
                    self.orange_memory[id_obj] = {"scores": [], "decision": None, "last_seen": current_time}
                
                mem = self.orange_memory[id_obj]
                mem["last_seen"] = current_time
                
                if mem["decision"] is None:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        input_crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_NEAREST)
                        input_crop = cv2.cvtColor(input_crop, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
                        input_crop = np.expand_dims(input_crop, axis=0)

                        if self.model_type == "tflite":
                            self.classifier.set_tensor(self.input_details[0]['index'], input_crop)
                            self.classifier.invoke()
                            prediction = self.classifier.get_tensor(self.output_details[0]['index'])
                            raw_score = float(prediction[0][0])
                        else:
                            prediction = self.classifier.predict(input_crop, verbose=0)
                            raw_score = float(prediction[0][0])

                        mem["scores"].append(raw_score)

                        color = (0, 255, 0) if raw_score > 0.12 else (0, 0, 255)
                        temp_res = "MANIS" if raw_score > 0.12 else "ASAM"
                        progress = int((len(mem['scores'])/15)*100)
                        label_text = f"{temp_res} ({progress}%)"

                        if len(mem["scores"]) >= 15:
                            avg = sum(mem["scores"]) / len(mem["scores"])
                            mem["decision"] = "MANIS" if avg > 0.12 else "ASAM"
                    else:
                        label_text, color = "Mencari...", (255, 255, 255)
                
                else:
                    label_text = mem["decision"]
                    color = (0, 255, 0) if label_text == "MANIS" else (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

# --- 6. EKSEKUSI KAMERA ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="jeruk-app-final",
    video_transformer_factory=OrangeAnalyzer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": {"ideal": 640}},
        "audio": False
    },
    async_processing=True,
)
