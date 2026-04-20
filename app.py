import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        self.vignette_mask = None

    def create_vignette(self, rows, cols):
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return mask

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols = img.shape[:2]

        if self.vignette_mask is None or self.vignette_mask.shape[:2] != (rows, cols):
            self.vignette_mask = self.create_vignette(rows, cols)

        if self.mode == "Gece Görüşü (Yeşil)":
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Yeşil fosfor efekti
            img = cv2.merge([np.zeros_like(gray), gray, np.zeros_like(gray)])
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Termal (Gelişmiş)":
            # 1. Görüntüyü griye çevir ve kontrastı artır
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 2. Isı yayan kenarları belirginleştir (Edge Enhancement)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None)
            # 3. Ironbow Renk Paleti Uygula
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            # 4. Kenarları bindir (Isı imzası netliği için)
            img = cv2.addWeighted(thermal, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Kızılötesi (B&W)":
            # Askeri 'White Hot' modu
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
            img = (img * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)

        elif self.mode == "Ultra Parlak":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adaptif histogram eşitleme (Görüntü kalitesini bozmadan aydınlatır)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(gray)
            img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

        return img

# --- UI ---
st.title("🪖 Tactical Vision HD + Termal")

mode = st.selectbox(
    "Görüş Modu Seçin",
    ["Normal", "Gece Görüşü (Yeşil)", "Termal (Gelişmiş)", "Kızılötesi (B&W)", "Ultra Parlak"]
)

ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=TacticalTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1920}, # Full HD zorlama
            "height": {"ideal": 1080},
            "facingMode": "environment"
        },
        "audio": False
    },
    async_processing=True,
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode

st.sidebar.subheader("Sistem Durumu")
st.sidebar.write("✅ HD Stream Aktif")
st.sidebar.write(f"✅ Mod: {mode}")
