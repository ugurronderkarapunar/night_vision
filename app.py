import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import random
import requests

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# Termal ve kızılötesi efektleri
# -------------------------------
class ThermalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Termal"
        self.distance = 50.0   # metre
        self.location = "İstanbul, Beşiktaş"  # varsayılan
        self.face_detected = False

    def apply_thermal(self, img):
        """Görüntüyü termal (sıcak-soğuk) renklere çevir"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Kontrast artır
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Renk haritası (JET = mavi(soğuk) -> kırmızı(sıcak))
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        return thermal

    def apply_ir(self, img):
        """Kızılötesi benzeri (beyaz-sıcak, siyah-soğuk)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Invert edilmiş HOT (beyaz-sıcak)
        ir = cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)
        return ir

    def apply_night_vision(self, img):
        """Gece görüşü (yeşil tonları)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        night = cv2.merge([np.zeros_like(enhanced), enhanced, np.zeros_like(enhanced)])
        return night

    def draw_hud(self, img):
        h, w = img.shape[:2]
        green = (0, 255, 0)
        red = (0, 0, 255)
        # Merkez crosshair
        cx, cy = w//2, h//2
        cv2.line(img, (cx-30, cy), (cx-10, cy), green, 2)
        cv2.line(img, (cx+10, cy), (cx+30, cy), green, 2)
        cv2.line(img, (cx, cy-30), (cx, cy-10), green, 2)
        cv2.line(img, (cx, cy+10), (cx, cy+30), green, 2)
        cv2.circle(img, (cx, cy), 2, green, -1)
        # Mesafe ve konum
        cv2.putText(img, f"RNG: {self.distance:.1f} m", (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        cv2.putText(img, f"LOC: {self.location}", (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 1)
        cv2.putText(img, f"MODE: {self.mode.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        return img

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mode == "Termal":
            processed = self.apply_thermal(img)
        elif self.mode == "Kızılötesi":
            processed = self.apply_ir(img)
        elif self.mode == "Gece Görüşü":
            processed = self.apply_night_vision(img)
        else:
            processed = img
        processed = self.draw_hud(processed)
        return processed

# -------------------------------
# Konum alma (IP tabanlı - basit ve çalışır)
# -------------------------------
def get_location_by_ip():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        city = data.get('city', 'Bilinmiyor')
        region = data.get('region', '')
        return f"{city}, {region}" if region else city
    except:
        return "Konum alınamadı"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD - Gerçek Zamanlı Termal & Kızılötesi")

# Sidebar
with st.sidebar:
    st.header("Ayarlar")
    mode = st.selectbox("Görüş Modu", ["Termal", "Kızılötesi", "Gece Görüşü", "Normal"])
    
    st.subheader("Mesafe (Lazer Telemetre)")
    distance = st.slider("Mesafe (metre)", min_value=0.0, max_value=2000.0, value=245.0, step=5.0)
    
    st.subheader("Konum Bilgisi")
    location_option = st.radio("Konum kaynağı", ["Otomatik (IP)", "Manuel"])
    if location_option == "Manuel":
        city = st.text_input("Şehir", "İstanbul")
        district = st.text_input("İlçe", "Beşiktaş")
        location = f"{city}, {district}"
    else:
        location = get_location_by_ip()
        st.info(f"Tespit edilen konum: {location}")
    
    st.divider()
    st.caption("Not: Termal ve kızılötesi efektleri görüntü parlaklığına göre renklendirme yapar. Mesafe manuel olarak ayarlanabilir.")

# Video stream
ctx = webrtc_streamer(
    key="thermal-cam",
    video_transformer_factory=ThermalTransformer,
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
    ctx.video_transformer.distance = distance
    ctx.video_transformer.location = location

st.caption("💡 İpucu: Termal modda sıcak nesneler (yüz, ampul) kırmızı/turuncu, soğuk nesneler mavi görünür. Kızılötesi modda ise beyaz-sıcak, siyah-soğuk simüle edilir.")
