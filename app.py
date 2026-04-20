import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import random

st.set_page_config(page_title="MIL-SPEC Night Vision", layout="wide")

class MilitaryVisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"

    def add_noise(self, image):
        """Askeri cihazlardaki analog karıncalanma efekti."""
        noise = np.zeros(image.shape, np.int8)
        cv2.randn(noise, 0, 20)
        return cv2.add(image, noise, dtype=cv2.CV_8UC3)

    def apply_vignette(self, image):
        """Lens kenarlarındaki kararma efekti."""
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        vignette = np.copy(image)
        for i in range(3):
            vignette[:, :, i] = vignette[:, :, i] * mask
        return vignette

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.mode == "Gece Görüşü (Yeşil)":
            # 1. Parlaklığı artır (Işık yükseltici tüp simülasyonu)
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
            # 2. Gri tonlamaya çevir ve yeşil kanala bindir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            green_img = np.zeros_like(img)
            green_img[:, :, 1] = gray # Sadece Yeşil kanal
            # 3. Gerçekçilik efektleri
            img = self.add_noise(green_img)
            img = self.apply_vignette(img)
            # 4. Tarama çizgileri
            img[::4, :, :] = img[::4, :, :] * 0.9 

        elif self.mode == "Kızılötesi (Simüle)":
            # Askeri termal kameralar (FLIR) genellikle 'White Hot' veya 'Ironbow' kullanır
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Keskinliği artır (Termal kenar belirginleştirme)
            gray = cv2.detailEnhance(gray, sigma_s=10, sigma_r=0.15)
            # Isı haritası uygula (COLORMAP_IRONBOW askeri standartlara yakındır)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_BONE) # Alternatif: COLORMAP_JET
            img = self.apply_vignette(thermal)
            # Digital HUD efekti için hafif noise
            img = self.add_noise(img)

        elif self.mode == "Ultra Parlak":
            # Siyah beyaz yüksek kontrastlı mod
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

        return img

# --- UI TASARIMI ---
st.title("🪖 MIL-SPEC Tactical Vision")

mode = st.select_slider(
    "Vizör Modu",
    options=["Normal", "Gece Görüşü (Yeşil)", "Ultra Parlak", "Kızılötesi (Simüle)"]
)

ctx = webrtc_streamer(
    key="military-vision",
    video_transformer_factory=MilitaryVisionTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"facingMode": "environment"},
        "audio": False
    },
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode

# Teknik Detaylar Paneli
col1, col2 = st.columns(2)
with col1:
    st.info("🛰️ **Sinyal Durumu:** Aktif\n\n🛡️ **Lens:** 35mm Gen3 Simülasyonu")
with col2:
    st.success("🔋 **Batarya:** %98\n\n📡 **Bağlantı:** Şifreli (WebRTC)")
