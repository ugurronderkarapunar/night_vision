import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        self.vignette_mask = None

    def create_vignette(self, rows, cols):
        kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
        kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return mask

    def draw_military_hud(self, frame):
        """Ekrana askeri hedefleme ve bilgi arayüzü çizer."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0) # Taktiksel Yeşil
        
        # Merkez Nişangah (Crosshair)
        cv2.line(frame, (cx - 40, cy), (cx - 10, cy), color, 2)
        cv2.line(frame, (cx + 10, cy), (cx + 40, cy), color, 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy - 10), color, 2)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 40), color, 2)
        cv2.circle(frame, (cx, cy), 2, color, -1)

        # Yan köşebentler (Targeting brackets)
        cv2.line(frame, (cx - 150, cy - 150), (cx - 100, cy - 150), color, 2)
        cv2.line(frame, (cx - 150, cy - 150), (cx - 150, cy - 100), color, 2)
        
        cv2.line(frame, (cx + 150, cy - 150), (cx + 100, cy - 150), color, 2)
        cv2.line(frame, (cx + 150, cy - 150), (cx + 150, cy - 100), color, 2)

        # Yanıp sönen REC ibaresi (Saniyeye göre)
        if int(time.time()) % 2 == 0:
            cv2.putText(frame, "• REC", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Sol Alt Sistem Bilgisi
        cv2.putText(frame, "FLIR SYS / WHT-HOT", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "ZOOM: 1.0x", (40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Sağ Üst Koordinat ve Sensör Verisi
        cv2.putText(frame, "LAT: 39.92 N", (w - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, "LON: 32.85 E", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, "ALT: 850M MSL", (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Sağ Alt Durum Verisi
        cv2.putText(frame, "BAT: [|||| ]", (w - 200, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "TRK: AUTO", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols = img.shape[:2]

        if self.vignette_mask is None or self.vignette_mask.shape[:2] != (rows, cols):
            self.vignette_mask = self.create_vignette(rows, cols)

        if self.mode == "Termal (Askeri FLIR)":
            # 1. Görüntüyü griye çevir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. White-Hot Efekti (Işık yayan yerleri aşırı parlat, diğer yerleri karart)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            thermal_core = clahe.apply(gray)
            
            # 3. Askeri kameralardaki sensör karıncalanması (Noise)
            noise = np.zeros(thermal_core.shape, np.int8)
            cv2.randn(noise, 0, 12)
            thermal_core = cv2.add(thermal_core, noise, dtype=cv2.CV_8UC1)
            
            # 4. Vignette uygula (Kenar karartması)
            thermal_core = (thermal_core * self.vignette_mask).astype(np.uint8)
            
            # 5. HUD çizmek için tekrar renkli formata geç (Görüntü siyah-beyaz kalır ama yeşil yazı yazabiliriz)
            img = cv2.cvtColor(thermal_core, cv2.COLOR_GRAY2BGR)
            img = self.draw_military_hud(img)

        elif self.mode == "Gece Görüşü (Yeşil)":
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([np.zeros_like(gray), gray, np.zeros_like(gray)])
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Ultra Parlak":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

        return img

# --- UI ---
st.title("🪖 Tactical Vision HD")

mode = st.selectbox(
    "Görüş Modu Seçin",
    ["Normal", "Termal (Askeri FLIR)", "Gece Görüşü (Yeşil)", "Ultra Parlak"]
)

ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=TacticalTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920},
            "height": {"ideal": 1080},
            "facingMode": "environment"
        },
        "audio": False
    },
    async_processing=True,
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode
