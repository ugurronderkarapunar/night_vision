import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import requests
import json

st.set_page_config(page_title="Tactical Vision HD", layout="wide")

# -------------------------------
# Yardımcı fonksiyonlar
# -------------------------------
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def estimate_distance(face_width_px, known_face_width_cm=16, focal_length_px=700):
    """Yüz genişliğinden tahmini mesafe (metre)"""
    if face_width_px <= 0:
        return None
    distance_cm = (known_face_width_cm * focal_length_px) / face_width_px
    return round(distance_cm / 100.0, 1)

def get_location_from_browser():
    """Tarayıcının Geolocation API'sini kullanarak konum alır (şehir/ilçe)"""
    import streamlit.components.v1 as components
    import time

    # Bileşen oluştur
    location_holder = st.empty()
    result = st.session_state.get("browser_location", None)

    if result is None:
        # JavaScript ile konum al
        components.html(
            """
            <script>
            function getLocation() {
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        function(position) {
                            const lat = position.coords.latitude;
                            const lon = position.coords.longitude;
                            const acc = position.coords.accuracy;
                            // Veriyi Streamlit'e gönder
                            const data = {lat: lat, lon: lon, acc: acc};
                            window.parent.postMessage({type: 'streamlit:setComponentValue', value: JSON.stringify(data)}, '*');
                        },
                        function(error) {
                            window.parent.postMessage({type: 'streamlit:setComponentValue', value: JSON.stringify({error: error.message})}, '*');
                        }
                    );
                } else {
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: JSON.stringify({error: "Geolocation not supported"})}, '*');
                }
            }
            getLocation();
            </script>
            <div>Konum alınıyor...</div>
            """,
            height=0,
        )
        # Bekle (basit bir bekleme, gerçek uygulamada daha karmaşık)
        time.sleep(2)
        # Bu yöntem streamlit-webrtc ile uyumlu değil, alternatif: session_state'e manuel atama
        st.warning("Tarayıcı konumu alınamadı. Lütfen manuel girin veya sayfayı yenileyip izin verin.")
        return "Konum alınamadı"

    # Eğer session_state'de varsa
    try:
        data = json.loads(result)
        if "error" in data:
            return f"Hata: {data['error']}"
        lat, lon = data["lat"], data["lon"]
        # Reverse geocode
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        headers = {"User-Agent": "TacticalVision/1.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        addr = resp.json()
        city = addr.get("address", {}).get("city") or addr.get("address", {}).get("town") or addr.get("address", {}).get("province") or "Bilinmiyor"
        district = addr.get("address", {}).get("suburb") or addr.get("address", {}).get("district") or ""
        return f"{city}, {district}" if district else city
    except:
        return "Dönüştürme hatası"

# -------------------------------
# Video işleme sınıfı
# -------------------------------
class TacticalTransformer(VideoTransformerBase):
    def __init__(self):
        self.mode = "Termal"
        self.distance = 0.0
        self.face_detected = False
        self.face_cascade = load_face_cascade()
        self.location_text = "Konum bekleniyor..."
        self.vignette = None

    def create_vignette(self, h, w):
        kernel_x = cv2.getGaussianKernel(w, w/2.5)
        kernel_y = cv2.getGaussianKernel(h, h/2.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        return (1 - 0.6) + mask * 0.6

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Vignette
        if self.vignette is None or self.vignette.shape != (h, w):
            self.vignette = self.create_vignette(h, w)

        # Yüz algılama (her 5 karede bir)
        if int(time.time() * 10) % 5 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            self.face_detected = len(faces) > 0
            if self.face_detected:
                x, y, fw, fh = faces[0]
                self.distance = estimate_distance(fw)
                if self.distance is None:
                    self.distance = round(np.random.uniform(0.5, 3.0), 1)
            else:
                # Yüz yoksa arka plan mesafesi (rastgele 5-50m)
                self.distance = round(np.random.uniform(5, 50), 1)

        # Termal simülasyonu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        if self.face_detected:
            # Sıcak renk (kırmızı-sarı)
            thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)
        else:
            # Soğuk renk (mavi-yeşil)
            thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_COOL)
        # Vignette ve gren
        thermal = (thermal * self.vignette[:, :, np.newaxis]).astype(np.uint8)
        noise = np.random.randint(0, 15, thermal.shape, dtype=np.uint8)
        thermal = cv2.add(thermal, noise)

        # HUD çiz
        self.draw_hud(thermal)
        return thermal

    def draw_hud(self, img):
        h, w = img.shape[:2]
        cx, cy = w//2, h//2
        green = (0,255,0)
        red = (0,0,255)
        # Merkez nişangah
        cv2.line(img, (cx-40, cy), (cx-10, cy), green, 2)
        cv2.line(img, (cx+10, cy), (cx+40, cy), green, 2)
        cv2.line(img, (cx, cy-40), (cx, cy-10), green, 2)
        cv2.line(img, (cx, cy+10), (cx, cy+40), green, 2)
        cv2.circle(img, (cx, cy), 2, green, -1)
        # REC
        if int(time.time()) % 2 == 0:
            cv2.putText(img, "● REC", (40,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
        # Mesafe ve hedef durumu
        status = "LIVE" if self.face_detected else "COLD"
        cv2.putText(img, f"RNG: {self.distance} m", (40, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        cv2.putText(img, f"TRG: {status}", (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        # Konum (sağ üst)
        lines = self.location_text.split(',')
        y = 40
        for line in lines[:2]:
            cv2.putText(img, line.strip(), (w-250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1)
            y += 25
        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🪖 Tactical Vision HD (Düzeltilmiş)")
st.markdown("**Gerçek termal değil, simülasyondur. Mesafe tahminidir.**")

# Konum al (manuel veya otomatik)
if "location_text" not in st.session_state:
    st.session_state.location_text = "Konum alınıyor..."

# Otomatik konum butonu
if st.sidebar.button("📍 Otomatik Konumumu Bul"):
    try:
        # Basitçe IP ile değil, tarayıcı konumunu kullanmak için bir yöntem:
        # Streamlit'in geolocation bileşeni yok, ama kullanıcıdan manuel giriş alalım.
        st.warning("Tarayıcı konumuna erişim için https üzerinde çalışmalı ve izin vermelisiniz. Şimdilik manuel giriş yapın.")
        # Alternatif: IP ile yaklaşık şehir (ipinfo.io)
        resp = requests.get("https://ipinfo.io/json", timeout=5)
        data = resp.json()
        city = data.get("city", "Bilinmiyor")
        region = data.get("region", "")
        st.session_state.location_text = f"{city}, {region}" if region else city
        st.success(f"Konum: {st.session_state.location_text}")
    except:
        st.error("Konum alınamadı")

# Manuel konum girişi
manual_location = st.sidebar.text_input("Veya manuel şehir/ilçe", value=st.session_state.location_text)
if manual_location != st.session_state.location_text:
    st.session_state.location_text = manual_location

# Mod seçimi
mode = st.sidebar.selectbox("Görüş Modu", ["Termal", "Gece Görüşü", "Normal"])
st.sidebar.info("🔴 Yüz algılanırsa KIRMIZI (sıcak), algılanmazsa MAVİ (soğuk)")

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
    ctx.video_transformer.location_text = st.session_state.location_text

st.caption("💡 Not: Gerçek termal kamera donanımı olmadan sadece simülasyon yapılabilir. Mesafe yüz genişliğine göre tahminidir. Konum için IP tabanlı yaklaşık değer kullanılıyor. Daha doğru için manuel girin.")
