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
        font-size: 12px; color: #aaaaaa; margin-top: 15px; font-style: italic;
        border-top: 1px solid #444; padding-top: 8px;
    }
    @media (max-width: 768px) { .element-container iframe { height: 500px !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. DETEKSI DEVICE ---
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH')
is_mobile = screen_width is not None and screen_width < 768

# --- 3. LOAD MODELS (DIET RAM VERSION) ---
@st.cache_resource
def load_and_prepare_models():
    tflite_path = 'model_jeruk_siam.tflite'
    # Pastikan file yolov8n.pt ada di GitHub
    detector = YOLO('yolov8n.pt') 
    
    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        return detector, interpreter
    else:
        st.error("⚠️ File model_jeruk_siam.tflite TIDAK ditemukan di GitHub!")
        return detector, None

detector, classifier = load_and_prepare_models()

# --- 4. UI: PANDUAN PENGGUNA ---
st.title("🍊 Deteksi & Klasifikasi Kualitas Jeruk Siam")

panduan_text = "📱 Panduan Mobile" if is_mobile else "💻 Panduan PC/Laptop"
st.markdown(f"""
<div class="instruction-box">
    <h4>{panduan_text}:</h4>
    <ol>
        <li>Tekan tombol <b>START</b> untuk mengaktifkan kamera.</li>
        <li>Arahkan kamera ke buah jeruk (jarak 15-30 cm).</li>
        <li>Box akan berwarna <b>ORANYE</b> selama proses analisa (15 frame).</li>
        <li>Hasil akan <b>TERKUNCI</b> menjadi <b>HIJAU (MANIS)</b> atau <b>MERAH (ASAM)</b>.</li>
    </ol>
    <div class="tech-tag">Engine: YOLOv8 & MobileNetV2 (TFLite Optimized)</div>
</div>
""", unsafe_allow_html=True)

# --- 5. ENGINE ANALYZER (LOGIKA WARNA & LOCK HASIL) ---
class OrangeAnalyzer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector
        self.classifier = classifier
        self.orange_memory = {}
        if classifier:
            self.input_details = classifier.get_input_details()
            self.output_details = classifier.get_output_details()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Hapus memori jeruk yang sudah tidak terlihat lebih dari 2 detik
        to_delete = [obj_id for obj_id, data in self.orange_memory.items() 
                     if current_time - data["last_seen"] > 2.0]
        for obj_id in to_delete: del self.orange_memory[obj_id]

        # Deteksi & Tracking (imgsz 256 agar RAM aman)
        results = self.detector.track(img, persist=True, conf=0.5, classes=[47, 49], verbose=False, imgsz=256)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id_obj in zip(boxes, ids):
                x1, y1, x2, y2 = box
                if id_obj not in self.orange_memory:
                    self.orange_memory[id_obj] = {"scores": [], "decision": None, "last_seen": current_time}
                
                mem = self.orange_memory[id_obj]
                mem["last_seen"] = current_time
                
                # JIKA BELUM LOCK (PROSES ANALISIS)
                if mem["decision"] is None:
                    current_color = (0, 165, 255) # Oranye (BGR)
                    
                    if self.classifier:
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            # Preprocessing TFLite
                            input_crop = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
                            input_crop = np.expand_dims(input_crop, axis=0)
                            
                            self.classifier.set_tensor(self.input_details[0]['index'], input_crop)
                            self.classifier.invoke()
                            raw_score = float(self.classifier.get_tensor(self.output_details[0]['index'])[0][0])
                            
                            mem["scores"].append(raw_score)
                        
                        progress = int((len(mem['scores'])/15)*100)
                        label_text = f"Menganalisa... {progress}%"

                        if len(mem["scores"]) >= 15:
                            avg = sum(mem["scores"]) / 15
                            mem["decision"] = "MANIS" if avg > 0.12 else "ASAM"
                
                # JIKA SUDAH LOCK (HASIL AKHIR)
                else:
                    label_text = f"HASIL: {mem['decision']}"
                    current_color = (0, 255, 0) if mem["decision"] == "MANIS" else (0, 0, 255)

                # Gambar Bounding Box Kustom
                cv2.rectangle(img, (x1, y1), (x2, y2), current_color, 3)
                # Background Label
                cv2.rectangle(img, (x1, y1 - 35), (x2, y1), current_color, -1)
                cv2.putText(img, label_text, (x1 + 5, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img

# --- 6. EKSEKUSI ---
webrtc_streamer(
    key="jeruk-siam-final",
    video_transformer_factory=OrangeAnalyzer,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": {"facingMode": "environment", "width": {"ideal": 640}}, "audio": False},
    async_processing=True,
)