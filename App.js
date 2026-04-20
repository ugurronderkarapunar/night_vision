import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Slider, SafeAreaView } from 'react-native';
import { Camera } from 'expo-camera';
import * as MediaLibrary from 'expo-media-library';

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [mode, setMode] = useState('normal'); // 'normal', 'green', 'ultra'
  const [zoom, setZoom] = useState(0);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      const mediaStatus = await MediaLibrary.requestPermissionsAsync();
      setHasPermission(status === 'granted' && mediaStatus.status === 'granted');
    })();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      await MediaLibrary.saveToLibraryAsync(photo.uri);
      alert('Fotoğraf Galeriye Kaydedildi!');
    }
  };

  if (hasPermission === null) return <View />;
  if (hasPermission === false) return <Text>Kameraya erişim izni verilmedi.</Text>;

  // Modlara göre stil ayarları
  const getOverlayStyle = () => {
    switch (mode) {
      case 'green':
        return { backgroundColor: 'rgba(0, 255, 0, 0.15)', opacity: 0.5 };
      case 'ultra':
        return { backgroundColor: 'rgba(255, 255, 255, 0.2)', opacity: 0.4 };
      default:
        return { backgroundColor: 'transparent' };
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <Camera 
        style={styles.camera} 
        type={type} 
        zoom={zoom}
        ref={cameraRef}
        whiteBalance="fluorescent" // Düşük ışıkta daha iyi sonuç verir
        exposure={1} // Pozlamayı artırır
      >
        {/* Yazılımsal Filtre Katmanı */}
        <View style={[StyleSheet.absoluteFill, getOverlayStyle()]} />

        {/* UI Paneli */}
        <View style={styles.buttonContainer}>
          <View style={styles.modeSelector}>
            <TouchableOpacity onPress={() => setMode('normal')} style={[styles.btn, mode === 'normal' && styles.activeBtn]}>
              <Text style={styles.text}>Normal</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setMode('green')} style={[styles.btn, mode === 'green' && styles.activeBtn]}>
              <Text style={styles.text}>Gece Görüşü</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setMode('ultra')} style={[styles.btn, mode === 'ultra' && styles.activeBtn]}>
              <Text style={styles.text}>Ultra Parlak</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.controls}>
             <Text style={styles.text}>Zoom</Text>
             <Slider
                style={{width: 200, height: 40}}
                minimumValue={0}
                maximumValue={1}
                onValueChange={(val) => setZoom(val)}
                minimumTrackTintColor="#00FF00"
                maximumTrackTintColor="#ffffff"
              />
          </View>

          <TouchableOpacity style={styles.captureBtn} onPress={takePicture}>
            <View style={styles.innerCaptureBtn} />
          </TouchableOpacity>
          
          <Text style={styles.proMode}>Pro Mode (Yakında)</Text>
        </View>
      </Camera>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'column',
    justifyContent: 'flex-end',
    alignItems: 'center',
    marginBottom: 20
  },
  modeSelector: {
    flexDirection: 'row',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    padding: 10,
    marginBottom: 20
  },
  btn: { padding: 10, marginHorizontal: 5 },
  activeBtn: { borderBottomWidth: 2, borderBottomColor: '#00FF00' },
  text: { color: 'white', fontWeight: 'bold' },
  captureBtn: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10
  },
  innerCaptureBtn: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'white'
  },
  proMode: { color: '#555', fontSize: 12, marginTop: 10 },
  controls: { alignItems: 'center', marginBottom: 20 }
});
