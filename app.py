import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# Video işleme sınıfı (Termal + Gece Görüşü)
# -------------------------------
class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Termal"          # "Termal" veya "Gece Görüşü"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.vignette = None
        self.face_region = None        # Son algılanan yüz koordinatları

    def create_vignette(self, h, w):
        """Köşeleri karartan efekt"""
        kernel_x = cv2.getGaussianKernel(w, w/2.5)
        kernel_y = cv2.getGaussianKernel(h, h/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return (1 - 0.5) + mask * 0.5   # Daha hafif vignette

    def apply_thermal_effect(self, img):
        """Gerçekçi termal kamera simülasyonu"""
        h, w = img.shape[:2]

        # Yüz algılama (her 10 karede bir)
        if int(time.time() * 10) % 10 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                self.face_region = faces[0]  # (x,y,w,h)
            else:
                self.face_region = None

        # Önce gri tonlamaya çevir ve kontrast artır
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Termal renk haritası oluştur (mavi->mor->kırmızı->sarı)
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)  # Daha gerçekçi

        # Yüz bölgesini daha sıcak (kırmızıya kaydır)
        if self.face_region is not None:
            x, y, fw, fh = self.face_region
            # Yüz bölgesinde renkleri kırmızı/sarı yap
            face_roi = thermal[y:y+fh, x:x+fw]
            # Parlaklığı ve kırmızı kanalı artır
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)  # Doygunluk artışı
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  # Parlaklık artışı
            face_roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # Kırmızı tonu baskın hale getir
            face_roi[:,:,2] = np.clip(face_roi[:,:,2] * 1.4, 0, 255)  # R kanalı
            thermal[y:y+fh, x:x+fw] = face_roi

        # Vignette ekle
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        thermal = (thermal * self.vignette[:, :, np.newaxis]).astype(np.uint8)

        # Gerçekçi gren efekti (düşük yoğunluklu)
        noise = np.random.randint(0, 10, thermal.shape, dtype=np.uint8)
        thermal = cv2.add(thermal, noise)

        # Hafif bulanıklık (termal kameralar biraz yumuşak)
        thermal = cv2.GaussianBlur(thermal, (3, 3), 0.5)

        return thermal

    def apply_night_vision_effect(self, img):
        """Gece görüşü simülasyonu (yeşil tonlamalı, grenli, düşük ışık)"""
        h, w = img.shape[:2]

        # Önce renkleri yeşil kanala kaydır
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Kontrastı artır
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Yeşil tonlama
        night = cv2.merge([enhanced, enhanced, enhanced])  # Gri
        night[:,:,1] = enhanced  # Yeşil kanalı koru
        night[:,:,0] = enhanced // 2  # Mavi azalt
        night[:,:,2] = enhanced // 2  # Kırmızı azalt

        # Işık seviyesini düşür
        night = (night * 0.7).astype(np.uint8)

        # Vignette
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        night = (night * self.vignette[:, :, np.newaxis]).astype(np.uint8)

        # Gren (gece görüşüne özel daha belirgin)
        noise = np.random.randint(0, 25, night.shape, dtype=np.uint8)
        night = cv2.add(night, noise)

        # Çizgi tarama efekti (opsiyonel)
        for i in range(0, h, 4):
            night[i:i+2, :] = night[i:i+2, :] * 0.8

        return night

    def draw_hud(self, img):
        """Basit HUD (sadece mod ve kayıt göstergesi)"""
        h, w = img.shape[:2]
        green = (0, 255, 0)
        red = (0, 0, 255)

        # Mod adı
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)

        # REC yanıp sönen
        if int(time.time()) % 2 == 0:
            cv2.putText(img, "● REC", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)

        return img

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.mode == "Termal":
            processed = self.apply_thermal_effect(img)
        elif self.mode == "Gece Görüşü":
            processed = self.apply_night_vision_effect(img)
        else:   # Normal mod
            processed = img

        processed = self.draw_hud(processed)
        return processed

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD (Gerçekçi Termal & Gece Görüşü)")
st.markdown("**Termal ve gece görüşü simülasyonudur. Yüz algılama ile sıcak bölgeler vurgulanır.**")

# Mod seçimi
mode = st.sidebar.selectbox("Görüş Modu", ["Termal", "Gece Görüşü", "Normal"])

# WebRTC streamer
ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=TacticalTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "environment"},
        "audio": False
    },
    async_processing=True,
)

if ctx.video_transformer:
    ctx.video_transformer.mode = mode

st.caption("💡 Gerçek termal donanım olmadan simülasyon yapılır. Yüz algılandığında termal modda kırmızı bölge belirginleşir. Gece görüşü yeşil tonlamalı ve grenlidir.")
