import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import random
import json
import requests

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# JavaScript ile coğrafi konum alma
# -------------------------------
def get_geolocation():
    """Tarayıcıdan konum almak için HTML/JS bileşeni"""
    geolocation_html = """
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const accuracy = position.coords.accuracy;
                const data = {lat: lat, lon: lon, acc: accuracy};
                const input = document.getElementById('geo_data');
                input.value = JSON.stringify(data);
                input.dispatchEvent(new Event('change'));
            }, function(error) {
                console.error(error);
                const input = document.getElementById('geo_data');
                input.value = JSON.stringify({error: error.message});
                input.dispatchEvent(new Event('change'));
            });
        } else {
            const input = document.getElementById('geo_data');
            input.value = JSON.stringify({error: "Geolocation not supported"});
            input.dispatchEvent(new Event('change'));
        }
    }
    getLocation();
    </script>
    <input type="hidden" id="geo_data" />
    """
    geo_data = st.components.v1.html(geolocation_html, height=0, width=0)
    # Streamlit'te bu şekilde direkt veri alamayız, bu yüzden alternatif olarak session_state'e kaydetmek için form kullanacağız.
    # Daha basit: kullanıcıdan manuel giriş veya varsayılan koordinat.
    # Ancak gerçek konum için st_geolocation paketi önerilir. Burada basit bir çözüm:
    return None

# -------------------------------
# Gerçek konum için alternatif (kullanıcı manuel girebilir veya varsayılan)
# -------------------------------
def get_default_location():
    return {"lat": 39.925533, "lon": 32.836287, "acc": 5}  # Ankara Kızılay

# -------------------------------
# Video işleme sınıfı
# -------------------------------
class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Normal"
        self.vignette_mask = None
        self.distance = 245.0          # metre cinsinden varsayılan mesafe
        self.last_distance_update = time.time()
        self.location = get_default_location()

    def create_vignette(self, rows, cols, strength=0.6):
        """Kenar karartma maskesi oluştur"""
        kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
        kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = (1 - strength) + mask * strength
        return mask

    def add_sensor_noise(self, img, intensity=0.05):
        """Sensör greni (karıncalanma) ekle"""
        noise = np.random.randn(*img.shape) * intensity * 255
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def apply_thermal_palette(self, gray):
        """Gri görüntüyü termal renk haritasına dönüştür"""
        # CLAHE ile kontrast artır
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Renk haritası (Inferno = sıcaklık hissi verir)
        thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        return thermal

    def draw_military_hud(self, frame, distance, location):
        """Gelişmiş askeri HUD çizer (nişangah, bilgiler, telemetre)"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 255, 0)  # Taktiksel yeşil
        red = (0, 0, 255)

        # Merkez Nişangah (Crosshair)
        cv2.line(frame, (cx - 40, cy), (cx - 10, cy), color, 2)
        cv2.line(frame, (cx + 10, cy), (cx + 40, cy), color, 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy - 10), color, 2)
        cv2.line(frame, (cx, cy + 10), (cx, cy + 40), color, 2)
        cv2.circle(frame, (cx, cy), 2, color, -1)

        # Yan köşebentler (Targeting brackets)
        bracket_size = 120
        cv2.line(frame, (cx - bracket_size, cy - bracket_size), (cx - bracket_size + 40, cy - bracket_size), color, 2)
        cv2.line(frame, (cx - bracket_size, cy - bracket_size), (cx - bracket_size, cy - bracket_size + 40), color, 2)
        cv2.line(frame, (cx + bracket_size, cy - bracket_size), (cx + bracket_size - 40, cy - bracket_size), color, 2)
        cv2.line(frame, (cx + bracket_size, cy - bracket_size), (cx + bracket_size, cy - bracket_size + 40), color, 2)

        # Yanıp sönen REC ibaresi
        if int(time.time()) % 2 == 0:
            cv2.putText(frame, "● REC", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)

        # Sol Alt Sistem Bilgisi
        cv2.putText(frame, f"FLIR SYS / {self.mode.upper()}", (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        cv2.putText(frame, "ZOOM: 1.0x", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        cv2.putText(frame, f"RNG: {distance:.1f} m", (40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Sağ Üst – Konum ve Sensör Verisi
        if location and "lat" in location:
            cv2.putText(frame, f"LAT: {location['lat']:.5f}", (w - 210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.putText(frame, f"LON: {location['lon']:.5f}", (w - 210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.putText(frame, f"ACC: {location.get('acc', 5)} m", (w - 210, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        else:
            cv2.putText(frame, "LOC: --", (w - 210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Sağ Alt – Batarya ve Durum
        cv2.putText(frame, "BAT: [||||| ]", (w - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "TRK: AUTO", (w - 200, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rows, cols = img.shape[:2]

        # Vignette maskesi oluştur (sadece termal/gece modları için)
        if self.mode in ["Termal (Askeri FLIR)", "Gece Görüşü (Yeşil)", "Kızılötesi (SWIR)"]:
            if self.vignette_mask is None or self.vignette_mask.shape[:2] != (rows, cols):
                self.vignette_mask = self.create_vignette(rows, cols, strength=0.7)

        # --- MODLAR ---
        if self.mode == "Termal (Askeri FLIR)":
            # 1. Gri tonlamaya çevir
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 2. Termal renk haritası uygula
            thermal = self.apply_thermal_palette(gray)
            # 3. Sensör gürültüsü ekle
            thermal = self.add_sensor_noise(thermal, intensity=0.03)
            # 4. Kenar karartma
            thermal = (thermal * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            # 5. HUD çiz
            img = self.draw_military_hud(thermal, self.distance, self.location)

        elif self.mode == "Gece Görüşü (Yeşil)":
            # Parlaklık artır
            bright = cv2.convertScaleAbs(img, alpha=1.8, beta=30)
            gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
            # CLAHE ile kontrast
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # Yeşil kanalı baskın, kırmızı/mavi sıfır
            green = np.zeros((rows, cols, 3), dtype=np.uint8)
            green[:, :, 1] = enhanced
            # Gren efekti
            green = self.add_sensor_noise(green, intensity=0.05)
            # Vignette
            green = (green * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            # HUD
            img = self.draw_military_hud(green, self.distance, self.location)

        elif self.mode == "Kızılötesi (SWIR)":
            # Kısa dalga kızılötesi simülasyonu (beyaz-sıcak + mavi-soğuk)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # Colormap JET (kırmızı=sıcak, mavi=soğuk)
            swir = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
            swir = self.add_sensor_noise(swir, intensity=0.02)
            swir = (swir * self.vignette_mask[:, :, np.newaxis]).astype(np.uint8)
            img = self.draw_military_hud(swir, self.distance, self.location)

        elif self.mode == "Ultra Parlak":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            bright = clahe.apply(gray)
            img = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
            # HUD yine de eklenebilir (opsiyonel)
            img = self.draw_military_hud(img, self.distance, self.location)

        else:  # Normal mod
            img = self.draw_military_hud(img, self.distance, self.location)

        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD")
st.markdown("**Gelişmiş termal, gece görüşü ve kızılötesi simülasyonu**")

# Kenar çubuğu kontrolleri
with st.sidebar:
    st.header("🎮 Kontroller")
    mode = st.selectbox(
        "Görüş Modu",
        ["Normal", "Termal (Askeri FLIR)", "Gece Görüşü (Yeşil)", "Kızılötesi (SWIR)", "Ultra Parlak"]
    )
    
    st.divider()
    st.subheader("📏 Lazer Telemetre")
    # Mesafe simülasyonu – buton ile rastgele gerçekçi mesafe
    if st.button("🔴 ÖLÇ (Laser RNG)"):
        # 20 ile 2000 metre arasında rastgele, ancak sürekli değişim yapay olmasın
        # Birkaç farklı senaryo: kısa (20-150), orta (150-600), uzun (600-2000)
        scenario = random.choices(["short", "mid", "long"], weights=[0.3, 0.5, 0.2])[0]
        if scenario == "short":
            new_dist = random.uniform(20, 150)
        elif scenario == "mid":
            new_dist = random.uniform(150, 600)
        else:
            new_dist = random.uniform(600, 2000)
        # Bir önceki mesafeye çok yakın olmaması için
        if "last_dist" in st.session_state:
            while abs(new_dist - st.session_state.last_dist) < 50 and len(st.session_state) > 0:
                new_dist = random.uniform(20, 2000)
        st.session_state.last_dist = new_dist
        st.session_state.distance = new_dist
        st.success(f"Mesafe: {new_dist:.1f} metre")
    
    # Manuel mesafe girişi
    manual_dist = st.number_input("Manuel mesafe (metre)", min_value=0.0, max_value=5000.0, value=st.session_state.get("distance", 245.0), step=10.0)
    if manual_dist != st.session_state.get("distance", 245.0):
        st.session_state.distance = manual_dist
    
    st.divider()
    st.subheader("📍 Konum Bilgisi")
    # Konum alma (basit: varsayılan veya manuel)
    loc_option = st.radio("Konum kaynağı", ["Varsayılan (Ankara)", "Manuel Giriş"])
    if loc_option == "Manuel Giriş":
        lat = st.number_input("Enlem", value=39.9255, format="%.6f")
        lon = st.number_input("Boylam", value=32.8363, format="%.6f")
        acc = st.number_input("Doğruluk (metre)", value=5, min_value=1)
        location = {"lat": lat, "lon": lon, "acc": acc}
    else:
        location = {"lat": 39.925533, "lon": 32.836287, "acc": 5}  # Ankara
    
    st.info(f"Kullanılan konum: {location['lat']:.5f}, {location['lon']:.5f}")

# Başlangıçta session_state mesafe değerini ayarla
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

if ctx.video_transformer:
    ctx.video_transformer.mode = mode
    ctx.video_transformer.distance = st.session_state.distance
    ctx.video_transformer.location = location
