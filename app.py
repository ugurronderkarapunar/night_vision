import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Sayfa Genişliği
st.set_page_config(page_title="MIL-SPEC High Def", layout="wide")

class MilitaryVisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        # Maskeleri bir kez oluşturup önbelleğe alarak performansı artırıyoruz
        self.vignette_mask = None

    def create_vignette(self, rows, cols):
        """Performans için maskeyi sadece çözünürlük değiştiğinde bir kez oluşturur."""
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return mask

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols = img.shape[:2]

        # Çözünürlük değişirse veya ilk kez çalışıyorsa maskeyi güncelle
        if self.vignette_mask is None or self.vignette_mask.shape[:2] != (rows, cols):
            self.vignette_mask = self.create_vignette(rows, cols)

        if self.mode == "Gece Görüşü (Yeşil)":
            # Hızlı Parlaklık Artırma
            img = cv2.convertScaleAbs(img, alpha=1.4, beta=40)
            # Yeşil Kanala Hızlı Dönüşüm
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([np.zeros_like(gray), gray, np.zeros_like(gray)])
            # Hızlı Vignette Uygulama
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Kızılötesi (Simüle)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Daha stabil bir termal efekt
            img = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Ultra Parlak":
            # Hızlı Histogram Eşitleme (Tüm görüntü yerine gri üzerinden)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return img

# --- UI ---
st.title("🪖 Tactical Vision HD")

# Mod seçimi - Butonlar yerine selectbox daha stabil geçiş sağlar
mode = st.selectbox(
    "Görüş Modu Seçin",
    ["Normal", "Gece Görüşü (Yeşil)", "Ultra Parlak", "Kızılötesi (Simüle)"]
)

# --- WebRTC Konfigürasyonu (KALİTE AYARI BURADA) ---
ctx = webrtc_streamer(
    key="high-def-vision",
    video_transformer_factory=MilitaryVisionTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            # Çözünürlüğü HD olarak zorluyoruz
            "width": {"ideal": 1280, "min": 800},
            "height": {"ideal": 720, "min": 600},
            "frameRate": {"ideal": 30}, # Akıcılık için 30 FPS
            "facingMode": "environment"
        },
        "audio": False
    },
    async_processing=True, # İşlemleri asenkron yaparak UI donmasını engeller
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode
