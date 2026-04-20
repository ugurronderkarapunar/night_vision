import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# Video işleme sınıfı (Termal + Gece Görüşü + Normal)
# -------------------------------
class TacticalTransformer(VideoTransformerBase):
    def __init__(self, mode_callback=None):
        self.mode = "Termal"           # varsayılan
        self.mode_callback = mode_callback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.vignette = None
        self.face_region = None

    def create_vignette(self, h, w):
        """Köşeleri karartan efekt (hafif)"""
        kernel_x = cv2.getGaussianKernel(w, w/2.5)
        kernel_y = cv2.getGaussianKernel(h, h/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return (1 - 0.5) + mask * 0.5

    def apply_thermal_effect(self, img):
        """Gerçekçi termal kamera simülasyonu"""
        h, w = img.shape[:2]

        # Yüz algılama (her 10 karede bir)
        if int(time.time() * 10) % 10 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            self.face_region = faces[0] if len(faces) > 0 else None

        # Gri tonlama ve kontrast artırma
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Termal renk haritası (inferno - gerçekçi)
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)

        # Yüz bölgesini daha sıcak (kırmızı/sarı) yap
        if self.face_region is not None:
            x, y, fw, fh = self.face_region
            face_roi = thermal[y:y+fh, x:x+fw]
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)  # Doygunluk
            hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  # Parlaklık
            face_roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            face_roi[:,:,2] = np.clip(face_roi[:,:,2] * 1.4, 0, 255)  # Kırmızı kanal
            thermal[y:y+fh, x:x+fw] = face_roi

        # Vignette
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        thermal = (thermal * self.vignette[:, :, np.newaxis]).astype(np.uint8)

        # Gren ve hafif bulanıklık
        noise = np.random.randint(0, 10, thermal.shape, dtype=np.uint8)
        thermal = cv2.add(thermal, noise)
        thermal = cv2.GaussianBlur(thermal, (3, 3), 0.5)

        return thermal

    def apply_night_vision_effect(self, img):
        """Gelişmiş gece görüşü simülasyonu (loş ışıkta çalışır)"""
        h, w = img.shape[:2]

        # Parlaklık analizi
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        # Çok karanlıksa gamma düzeltme ile aydınlat
        if avg_brightness < 40:
            gamma = 2.0
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
            # Kontrast artırma
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Gri tonlama ve CLAHE ile detayları ortaya çıkar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Yeşil tonlama
        night = cv2.merge([enhanced, enhanced, enhanced])
        night[:,:,1] = enhanced          # Yeşil kanal tam
        night[:,:,0] = enhanced // 2     # Mavi az
        night[:,:,2] = enhanced // 2     # Kırmızı az

        # Işık seviyesine göre parlaklık ayarı
        if avg_brightness < 40:
            night = np.clip(night * 1.3, 0, 255).astype(np.uint8)
        else:
            night = (night * 0.8).astype(np.uint8)

        # Vignette
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        night = (night * self.vignette[:, :, np.newaxis]).astype(np.uint8)

        # Gren efekti (karanlıkta daha belirgin)
        noise_level = 15 if avg_brightness < 40 else 8
        noise = np.random.randint(0, noise_level, night.shape, dtype=np.uint8)
        night = cv2.add(night, noise)

        # Yatay tarama çizgileri (gece görüşü karakteristiği)
        for i in range(0, h, 6):
            night[i:i+2, :] = night[i:i+2, :] * 0.85

        # Keskinleştirme
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        night = cv2.filter2D(night, -1, kernel)

        return night

    def draw_hud(self, img):
        """Basit HUD (sadece mod ve kayıt göstergesi)"""
        green = (0, 255, 0)
        red = (0, 0, 255)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        if int(time.time()) % 2 == 0:
            cv2.putText(img, "● REC", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
        return img

    def transform(self, frame):
        # Callback ile mod güncellemesi (anlık)
        if self.mode_callback and callable(self.mode_callback):
            new_mode = self.mode_callback()
            if new_mode != self.mode:
                self.mode = new_mode

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
st.markdown("**Termal ve gece görüşü simülasyonudur. Modu sol menüden anında değiştirebilirsiniz.**")

# Mod seçimi (canlı güncellenir)
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Termal"

mode = st.sidebar.selectbox(
    "Görüş Modu",
    ["Termal", "Gece Görüşü", "Normal"],
    index=["Termal", "Gece Görüşü", "Normal"].index(st.session_state.selected_mode)
)
st.session_state.selected_mode = mode

# Callback fonksiyonu (transformer her karede bunu çağırır)
def get_current_mode():
    return st.session_state.selected_mode

# WebRTC streamer
ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=lambda: TacticalTransformer(mode_callback=get_current_mode),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "environment"},
        "audio": False
    },
    async_processing=True,
)

st.caption("💡 **İpuçları:**\n"
           "- Termal modda yüz algılandığında kırmızı/sarı bölge belirginleşir.\n"
           "- Gece görüşü loş ortamlarda otomatik aydınlatma yapar.\n"
           "- Normal modda hiçbir efekt uygulanmaz.\n"
           "- Modu sol menüden değiştirdikten 1-2 saniye sonra efekt uygulanır.")
