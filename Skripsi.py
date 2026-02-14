import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Multi-Detection Jeruk", page_icon="🍊", layout="centered")
st.title("🍊 Multi-Object Detection & Stable Classification")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_all_models():
    try:
        classifier = tf.keras.models.load_model('model_jeruk_rgb_final.keras')
        # Gunakan model YOLO yang mendukung tracking
        detector = YOLO('yolov8n.pt') 
        return classifier, detector
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

classifier, detector = load_all_models()

# --- 3. DATABASE MEMORI OBJEK ---
# Struktur: { id_objek: {"scores": [], "decision": None, "last_seen": timestamp} }
if 'orange_memory' not in st.session_state:
    st.session_state.orange_memory = {}

# --- 4. FUNGSI PREDIKSI & STABILISASI ---
def get_stable_prediction(id_objek, image_crop):
    current_time = time.time()
    
    # Inisialisasi memori jika ID baru
    if id_objek not in st.session_state.orange_memory:
        st.session_state.orange_memory[id_objek] = {
            "scores": [],
            "decision": None,
            "last_seen": current_time
        }
    
    mem = st.session_state.orange_memory[id_objek]
    mem["last_seen"] = current_time # Update waktu terlihat
    
    # Jika sudah ada keputusan final, langsung kembalikan hasilnya
    if mem["decision"] is not None:
        return mem["decision"], True, len(mem["scores"])

    # Jika belum ada keputusan, lakukan prediksi
    img = cv2.resize(image_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = classifier.predict(img, verbose=0)
    raw_score = float(prediction[0][0])
    
    # Simpan ke history ID tersebut
    mem["scores"].append(raw_score)
    
    # Hitung rata-rata jika sudah mencapai batas (20 frame)
    if len(mem["scores"]) >= 20:
        avg_score = sum(mem["scores"]) / len(mem["scores"])
        mem["decision"] = "MANIS" if avg_score > 0.12 else "ASAM"
        return mem["decision"], True, 20
    
    # Belum cukup data
    label_sementara = "MANIS" if raw_score > 0.12 else "ASAM"
    return label_sementara, False, len(mem["scores"])

# --- 5. LOGIKA AUTO-RESET ---
def cleanup_memory():
    current_time = time.time()
    # Hapus objek yang sudah tidak terlihat lebih dari 3 detik
    to_delete = [obj_id for obj_id, data in st.session_state.orange_memory.items() 
                 if current_time - data["last_seen"] > 3.0]
    for obj_id in to_delete:
        del st.session_state.orange_memory[obj_id]

# --- 6. ANTARMUKA KAMERA ---
run_camera = st.sidebar.checkbox("Aktifkan Kamera", value=False)
FRAME_WINDOW = st.image([]) 

camera = cv2.VideoCapture(0)

if run_camera:
    while True:
        ret, frame = camera.read()
        if not ret: break

        cleanup_memory() # Jalankan pembersihan berkala
        display_frame = frame.copy()
        
        # Gunakan detector.track untuk mendapatkan ID konstan
        results = detector.track(frame, persist=True, conf=0.5, classes=[47, 49], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id_obj in zip(boxes, ids):
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    label, is_locked, count = get_stable_prediction(id_obj, crop)
                    
                    # Warna: Hijau jika manis, Merah jika asam
                    color = (0, 255, 0) if label == "MANIS" else (0, 0, 255)
                    
                    # Gambar Bounding Box & ID
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Info Status
                    status = "LOCKED" if is_locked else f"Analyzing {count}/20"
                    txt = f"ID:{id_obj} {label} ({status})"
                    
                    cv2.putText(display_frame, txt, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
else:
    camera.release()