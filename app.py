import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import random

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# Video işleme sınıfı (Tüm efektler burada)
# -------------------------------
class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        self.vignette_mask = None
        self.distance = 245.0
        self.location = {"lat": 39.925533, "lon": 32.836287, "acc": 5}
        self.thermal_palette = "JET (Mavi-Kırmızı)"   # Varsayılan palet

    def create_vignette(self, rows, cols, strength=0.7):
        """Kenar karartma maskesi (vignette) oluşturur"""
        kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
        kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = (1 - strength) + mask * strength
        return mask

    def add_sensor_noise(self, img, intensity=0.03):
        """Sensör greni (karıncalanma) ekler"""
        noise = np.random.randn(*img.shape) * intensity * 255
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def apply_thermal_palette(self, gray, palette_name):
        """Seçilen renk paletine göre termal görüntü oluşturur"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        if palette_name == "JET (Mavi-Kırmızı)":
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        elif palette_name == "HOT (Siyah-Kırmızı-Sarı)":
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)
        elif palette_name == "INFERNO (Sıcaklık)":
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        elif palette_name == "MAGMA (Mor-Turuncu)":
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_MAGMA)
        else:  # Varsayılan JET
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        return colored

    def draw_military_hud(self, frame, distance, location):
        """Askeri HUD (nişangah, bilgiler, telemetre) çizer"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0)      # Yeşil
        red = (0, 0, 255)
        yellow = (0, 255, 255)

        # Merkez nişangah (crosshair)
        cv2.line(frame, (cx - 40, cy), (cx - 10, cy), color, 2)
        cv2.line(frame, (cx + 10, cy), (cx + 40, cy), color, 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy - 10), color, 2)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 40), color, 2)
        cv2.circle(frame, (cx, cy), 2, color, -1)

        # Köşebentler (targeting brackets)
        bracket = 120
        cv2.line(frame, (cx - bracket, cy - bracket), (cx - bracket + 40, cy - bracket), color, 2)
        cv2.line(frame, (cx - bracket, cy - bracket), (cx - bracket, cy - bracket + 40), color, 2)
        cv2.line(frame, (cx + bracket, cy - bracket), (cx + bracket - 40, cy - bracket), color, 2)
        cv2.line(frame, (cx + bracket, cy - bracket), (cx + bracket, cy - bracket + 40), color, 2)

        # Yanıp sönen REC
        if int(time.time()) % 2 == 0:
            cv2.putText(frame, "● REC", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)

        # Sol alt sistem bilgisi
        cv2.putText(frame, f"FLIR SYS / {self.mode.upper()}", (40, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "ZOOM: 1.0x", (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, f"RNG: {distance:.1f} m", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"BAT: [||||| ]", (40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Sağ üst konum
        cv2.putText(frame, f"LAT: {location['lat']:.5f}", (w - 210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"LON: {location['lon']:.5f}", (w - 210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"ACC: {location['acc']} m", (w - 210, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Sağ alt durum
        cv2.putText(frame, "TRK: AUTO", (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "SYS: OK", (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols = img.shape[:2]

        # Vignette maskesi (sadece belirli modlarda)
        if self.mode in ["Termal (Askeri FLIR)", "Gece Görüşü (Yeşil)", "Kızılötesi (SWIR)"]:
            if self.vignette_mask is None or self.vignette_mask.shape[:2] != (rows, cols):
                self.vignette_mask = self.create_vignette(rows, cols, strength=0.7)

        # ------------------------- MODLAR -------------------------
        if self.mode == "Termal (Askeri FLIR)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thermal = self.apply_thermal_palette(gray, self.thermal_palette)
            thermal = self.add_sensor_noise(thermal, intensity=0.03)
            thermal = (thermal * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            img = self.draw_military_hud(thermal, self.distance, self.location)

        elif self.mode == "Gece Görüşü (Yeşil)":
            bright = cv2.convertScaleAbs(img, alpha=1.8, beta=30)
            gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            green = np.zeros((rows, cols, 3), dtype=np.uint8)
            green[:, :, 1] = enhanced
            green = self.add_sensor_noise(green, intensity=0.05)
            green = (green * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            img = self.draw_military_hud(green, self.distance, self.location)

        elif self.mode == "Kızılötesi (SWIR)":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 🔧 DÜZELTME: Sıcak nesnelerin parlak, soğukların koyu olması için
            # Eğer renkler ters geliyorsa (sıcak mavi, soğuk kırmızı) aşağıdaki satırı aktif edin:
            # enhanced = 255 - enhanced
            
            # Kullanıcının seçtiği paleti uygula
            swir = self.apply_thermal_palette(enhanced, self.thermal_palette)
            swir = self.add_sensor_noise(swir, intensity=0.02)
            swir = (swir * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            img = self.draw_military_hud(swir, self.distance, self.location)

        elif self.mode == "Ultra Parlak":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            bright = clahe.apply(gray)
            img = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
            img = self.draw_military_hud(img, self.distance, self.location)

        else:  # Normal mod
            img = self.draw_military_hud(img, self.distance, self.location)

        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD")
st.markdown("**Gelişmiş termal, gece görüşü ve kızılötesi simülasyonu** | Sıcak nesneler sıcak renklerde gösterilir")

# Sidebar kontrolleri
with st.sidebar:
    st.header("🎮 Kontroller")
    
    mode = st.selectbox(
        "Görüş Modu",
        ["Normal", "Termal (Askeri FLIR)", "Gece Görüşü (Yeşil)", "Kızılötesi (SWIR)", "Ultra Parlak"]
    )
    
    st.divider()
    st.subheader("🎨 Renk Paleti (Termal / Kızılötesi)")
    thermal_palette = st.selectbox(
        "Palet seçin",
        ["JET (Mavi-Kırmızı)", "HOT (Siyah-Kırmızı-Sarı)", "INFERNO (Sıcaklık)", "MAGMA (Mor-Turuncu)"]
    )
    
    st.divider()
    st.subheader("📏 Lazer Telemetre")
    
    if st.button("🔴 ÖLÇ (Laser RNG)"):
        scenario = random.choices(["short", "mid", "long"], weights=[0.3, 0.5, 0.2])[0]
        if scenario == "short":
            new_dist = random.uniform(20, 150)
        elif scenario == "mid":
            new_dist = random.uniform(150, 600)
        else:
            new_dist = random.uniform(600, 2000)
        if "last_dist" in st.session_state:
            while abs(new_dist - st.session_state.last_dist) < 50:
                new_dist = random.uniform(20, 2000)
        st.session_state.last_dist = new_dist
        st.session_state.distance = new_dist
        st.success(f"Mesafe: {new_dist:.1f} metre")
    
    manual_dist = st.number_input("Manuel mesafe (metre)", min_value=0.0, max_value=5000.0, 
                                   value=st.session_state.get("distance", 245.0), step=10.0)
    if manual_dist != st.session_state.get("distance", 245.0):
        st.session_state.distance = manual_dist
    
    st.divider()
    st.subheader("📍 Konum Bilgisi")
    loc_option = st.radio("Konum kaynağı", ["Varsayılan (Ankara)", "Manuel Giriş"])
    if loc_option == "Manuel Giriş":
        lat = st.number_input("Enlem", value=39.9255, format="%.6f")
        lon = st.number_input("Boylam", value=32.8363, format="%.6f")
        acc = st.number_input("Doğruluk (metre)", value=5, min_value=1)
        location = {"lat": lat, "lon": lon, "acc": acc}
    else:
        location = {"lat": 39.925533, "lon": 32.836287, "acc": 5}
    
    st.info(f"Kullanılan konum: {location['lat']:.5f}, {location['lon']:.5f}")

# Session state başlatma
if "distance" not in st.session_state:
    st.session_state.distance = 245.0

# WebRTC streamer
ctx = webrtc_streamer(
    key="tactical-vision",
    video_transformer_factory=TacticalTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "facingMode": "environment"
        },
        "audio": False
    },
    async_processing=True,
)

# Transformer'a parametreleri aktar
if ctx.video_transformer:
    ctx.video_transformer.mode = mode
    ctx.video_transformer.distance = st.session_state.distance
    ctx.video_transformer.location = location
    ctx.video_transformer.thermal_palette = thermal_palette

# Yardım metni
st.caption("💡 **İpucu:** Kızılötesi modunda renkler ters geliyorsa (sıcak nesne mavi), kod içindeki 'enhanced = 255 - enhanced' satırını aktif edin.")
