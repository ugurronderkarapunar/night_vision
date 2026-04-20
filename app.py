import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

st.set_page_config(page_title="Night Vision AI", layout="centered")

st.title("🌙 Gece Görüşü Kamerası")
st.write("Mod seçin ve kamerayı başlatın.")

# Filtre Seçenekleri
mode = st.radio("Mod Seçimi", ["Normal", "Gece Görüşü (Yeşil)", "Ultra Parlak"], horizontal=True)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.mode == "Gece Görüşü (Yeşil)":
            # Parlaklık ve Kontrast Artırma
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
            # Yeşil Kanala Odaklanma
            img[:, :, 0] = 0  # Mavi kanalı sıfırla
            img[:, :, 2] = 0  # Kırmızı kanalı sıfırla
            # Hafif bir blur ile noise azaltma
            img = cv2.medianBlur(img, 3)

        elif self.mode == "Ultra Parlak":
            # Yüksek Parlaklık (Düşük ışık için)
            img = cv2.convertScaleAbs(img, alpha=2.5, beta=100)
            # Siyah-Beyaz yaparak ışık hassasiyetini simüle etme
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

# Kamerayı Başlat
ctx = webrtc_streamer(
    key="night-vision",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode

st.info("💡 Not: Telefonunuzda arka kameranın açılması için 'facingMode: environment' ayarı eklenmiştir.")
