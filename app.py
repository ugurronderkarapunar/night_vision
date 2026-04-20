import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Night Vision AI Pro",
    page_icon="🌙",
    layout="centered"
)

# Arayüz Başlıkları
st.title("🌙 Gece Görüşü & IR Simülatörü")
st.markdown("""
    Bu uygulama, düşük ışık koşullarında görüntüyü iyileştirmek için 
    **OpenCV** kullanarak gerçek zamanlı görüntü işleme yapar.
""")

# Yan Menü (Sidebar) Kontrolleri
st.sidebar.header("⚙️ Görüntü Ayarları")
mode = st.sidebar.radio(
    "Görüş Modunu Seçin:",
    ["Normal", "Gece Görüşü (Yeşil)", "Ultra Parlak", "Kızılötesi (Simüle)"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Mod Açıklamaları:**")
if mode == "Gece Görüşü (Yeşil)":
    st.sidebar.info("Parlaklığı artırır ve yeşil spektruma odalar.")
elif mode == "Ultra Parlak":
    st.sidebar.info("Maksimum pozlama ve kontrast ile siyah-beyaz görüntü sağlar.")
elif mode == "Kızılötesi (Simüle)":
    st.sidebar.info("Isı haritası (Thermal) efektini simüle eder.")

# Görüntü İşleme Sınıfı
class NightVisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"

    def transform(self, frame):
        # Görüntüyü kare (frame) olarak al (BGR formatında)
        img = frame.to_ndarray(format="bgr24")

        if self.mode == "Gece Görüşü (Yeşil)":
            # 1. Parlaklık (beta) ve Kontrast (alpha) artırma
            img = cv2.convertScaleAbs(img, alpha=1.8, beta=40)
            # 2. Yeşil filtre uygula (Mavi ve Kırmızı kanalları baskıla)
            img[:, :, 0] = img[:, :, 0] * 0.1  # Mavi
            img[:, :, 2] = img[:, :, 2] * 0.1  # Kırmızı
            # 3. Kumlanmayı azaltmak için hafif blur
            img = cv2.GaussianBlur(img, (3, 3), 0)

        elif self.mode == "Ultra Parlak":
            # Görüntüyü gri tonlamaya çevir (daha fazla ışık hassasiyeti için)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Kontrastı aşırı artır (CLAHE - Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            bright = clahe.apply(gray)
            # Tekrar renkli formata dönüştür (Streamlit uyumu için)
            img = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
            # Parlaklığı ek olarak artır
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=60)

        elif self.mode == "Kızılötesi (Simüle)":
            # Görüntüyü griye çevir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Isı haritası uygula (HOT veya JET kullanılabilir)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
            img = thermal

        return img

# WebRTC (Kamera) Bileşeni
ctx = webrtc_streamer(
    key="night-vision-app",
    video_transformer_factory=NightVisionTransformer,
    # Google STUN sunucuları (Bağlantı kurmayı kolaylaştırır)
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    # Mobil cihazlar için arka kamerayı zorla
    media_stream_constraints={
        "video": {
            "facingMode": "environment",
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
    async_processing=True,
)

# Modu sınıfa aktar
if ctx.video_transformer:
    ctx.video_transformer.mode = mode

st.write("---")
st.caption("İş bilgisayarından geliştirilmiştir. Telefon tarayıcısından erişim için HTTPS gereklidir.")
