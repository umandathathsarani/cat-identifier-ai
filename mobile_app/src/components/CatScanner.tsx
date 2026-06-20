import React, { useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
import { 
  Camera as VisionCamera, 
  useCameraDevice, 
  useCameraPermission, 
  // Ensure your vision-camera version is 4.x.x
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { Worklets } from 'react-native-worklets-core';

// Cast VisionCamera to any to satisfy TypeScript when using frameProcessor prop
const Camera: any = VisionCamera;

export default function CatScanner() {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();

  // Load the model
  const model = useTensorflowModel(require('../../assets/models/model.tflite'), 'default' as any);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Simple worklet-style frame processor (avoid useFrameProcessor import mismatch)
  // The 'worklet' directive marks this function for use as a Vision Camera frame processor
  const frameProcessor = (frame: any) => {
    'worklet';
    if (model.state !== 'loaded') return;

    // Use the model to run inference directly on the frame
    // Note: Ensure your TFLite model accepts the raw Frame buffer
    try {
      const outputs = model.model.runSync([frame]);
      // Using console.log is allowed inside worklets in Vision Camera
      console.log(outputs);
    } catch (e) {
      // swallow errors in worklet
    }
  };

  if (!hasPermission) return <Text style={styles.centerText}>Requesting Camera Permission...</Text>;
  if (device == null) return <Text style={styles.centerText}>No Camera Found</Text>;
  if (model.state === 'loading') return <ActivityIndicator style={styles.centerText} size="large" color="#00ff00" />;

  return (
    <View style={StyleSheet.absoluteFill}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv" // Keep as yuv for Vision Camera v4
      />
      
      <View style={styles.overlay}>
        <Text style={styles.scanText}>Scanning for Cats...</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  centerText: { flex: 1, textAlign: 'center', textAlignVertical: 'center', fontSize: 18 },
  overlay: { position: 'absolute', bottom: 50, alignSelf: 'center', backgroundColor: 'rgba(0,0,0,0.7)', padding: 20, borderRadius: 15 },
  scanText: { color: 'white', fontSize: 18, fontWeight: 'bold' }
});