import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import math

st.set_page_config(page_title="OPERATOR HUD Gen-5", layout="wide")

class TacticalHUDTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        self.vignette_mask = None
        self.start_time = time.time()

    def apply_thermal_effects(self, gray):
        """Gerçekçi termal doku ve keskinlik."""
        # Isı yayılımını simüle etmek için hafif blur ve ardından keskinleştirme
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        return sharp

    def draw_hud(self, frame, mode_name):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0) if "Yeşil" in mode_name else (255, 255, 255)
        
        # --- MERKEZ NİŞANGAH ---
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
        
        # --- MESAFE ÖLÇER (Stadiametric) ---
        # Objelerin odağa uzaklığını simüle eden dinamik bir veri
        dist = 150.5 + math.sin(time.time()) * 5 # Hareketli mesafe simülasyonu
        cv2.putText(frame, f"RNG: {dist:.1f}m", (cx + 50, cy + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- KONUM VE YÖNELİM ---
        cv2.putText(frame, "POS: 41.0082 N, 28.9784 E", (50, h - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"AZM: {abs(int(math.degrees(math.sin(time.time()/2))*2)) }' NW", 
                    (50, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # --- SİSTEM DURUMU ---
        cv2.putText(frame, f"MODE: {mode_name}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Batarya ve Zaman
        elapsed = int(time.time() - self.start_time)
        cv2.putText(frame, f"OP_TIME: {elapsed}s", (w - 200, h - 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        if self.mode == "Termal (FLIR)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Yüksek ısı kontrastı (Adaptive)
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            thermal = clahe.apply(gray)
            thermal = self.apply_thermal_effects(thermal)
            # White-Hot görünümü
            img = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
            img = self.draw_hud(img, "THERMAL-IR")

        elif self.mode == "Gece Görüşü (G3)":
            # Yeşil fosfor simülasyonu
            img = cv2.convertScaleAbs(img, alpha=1.8, beta=30)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            green = np.zeros_like(img)
            green[:, :, 1] = gray # Sadece yeşil kanal
            # Scanlines ve Noise
            noise = np.random.normal(0, 15, (h, w, 3)).astype(np.uint8)
            img = cv2.add(green, noise)
            img = self.draw_hud(img, "NVG-GEN3")

        elif self.mode == "Kızılötesi (Digital)":
            # Düşük ışıkta dijital sensör artırımı
            img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=50)
            img = self.draw_hud(img, "DIGITAL-IR")

        return img

# --- UI ---
st.title("🪖 Advanced Operator Visor")

mode = st.radio("Sistem Seçimi", 
               ["Normal", "Termal (FLIR)", "Gece Görüşü (G3)", "Kızılötesi (Digital)"],
               horizontal=True)

webrtc_streamer(
    key="operator-v5",
    video_transformer_factory=TacticalHUDTransformer,
    media_stream_constraints={
        "video": {"width": 1280, "height": 720, "facingMode": "environment"},
        "audio": False
    },
    async_processing=True,
)

if ctx := st.session_state.get("operator-v5"):
    if ctx.video_transformer:
        ctx.video_transformer.mode = mode
