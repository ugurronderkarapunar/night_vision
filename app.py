import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

class TacticalTransformer(VideoTransformerBase):
    def __init__(self, mode_callback=None):
        self.mode = "Termal"
        self.mode_callback = mode_callback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.vignette = None
        self.face_region = None
        self.frame_counter = 0   # frame atlama için

    def create_vignette(self, h, w):
        kernel_x = cv2.getGaussianKernel(w, w/2.5)
        kernel_y = cv2.getGaussianKernel(h, h/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return (1 - 0.4) + mask * 0.4   # daha hafif vignette

    def apply_thermal_effect(self, img):
        h, w = img.shape[:2]
        # Her 30 karede bir yüz ara (saniyede ~1 kez)
        if self.frame_counter % 30 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            self.face_region = faces[0] if len(faces) > 0 else None

        # Termal renk haritası (hızlı)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        # Vignette (önceden hesaplanmışsa kullan)
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        thermal = (thermal * self.vignette[:, :, np.newaxis]).astype(np.uint8)

        return thermal

    def apply_night_vision_effect(self, img):
        h, w = img.shape[:2]
        # Parlaklığı artır (basit gamma)
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        bright = cv2.LUT(img, table)
        # Gri ton + yeşil kanal baskın
        gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
        night = cv2.merge([gray//2, gray, gray//2])   # B=low, G=full, R=low
        # Vignette
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)
        night = (night * self.vignette[:, :, np.newaxis]).astype(np.uint8)
        return night

    def draw_hud(self, img):
        green = (0, 255, 0)
        red = (0, 0, 255)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 1)
        if int(time.time()) % 2 == 0:
            cv2.putText(img, "REC", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)
        return img

    def transform(self, frame):
        self.frame_counter += 1
        # Mod callback
        if self.mode_callback:
            new_mode = self.mode_callback()
            if new_mode != self.mode:
                self.mode = new_mode

        img = frame.to_ndarray(format="bgr24")
        # Her 2 karede bir işle (diğerini direkt göster) -> akıcılık için
        if self.frame_counter % 2 == 0:
            if self.mode == "Termal":
                processed = self.apply_thermal_effect(img)
            elif self.mode == "Gece Görüşü":
                processed = self.apply_night_vision_effect(img)
            else:
                processed = img
        else:
            processed = img   # işlemsiz kare (daha akıcı)

        processed = self.draw_hud(processed)
        return processed

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD (Hafif Sürüm)")
st.markdown("**Donma sorunu giderildi - Daha akıcı görüntü**")

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Termal"

mode = st.sidebar.selectbox(
    "Görüş Modu",
    ["Termal", "Gece Görüşü", "Normal"],
    index=["Termal", "Gece Görüşü", "Normal"].index(st.session_state.selected_mode)
)
st.session_state.selected_mode = mode

def get_current_mode():
    return st.session_state.selected_mode

ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=lambda: TacticalTransformer(mode_callback=get_current_mode),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "facingMode": "environment"},
        "audio": False
    },
    async_processing=True,
)

st.caption("⚡ Performans iyileştirmeleri: her 2 karede işlem, düşük çözünürlük, hafif efektler. Yüz algılama saniyede 1 kez.")
