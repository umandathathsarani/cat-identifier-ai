import React, { useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
// Notice the new 'useCameraPermission' hook imported here
import { Camera, useCameraDevice, useFrameProcessor, useCameraPermission } from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';

export default function CatScanner() {
  const device = useCameraDevice('back');
  
  // Modern Vision Camera permission handling
  const { hasPermission, requestPermission } = useCameraPermission();

  // 1. Load the TFLite Model from assets
  const model = useTensorflowModel(require('../../assets/models/model.tflite'));
  
  // 2. Initialize the high-performance resizer
  const { resize } = useResizePlugin();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // 3. The Frame Processor (Runs 30-60 times a second)
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    if (model.state !== 'loaded') return;

    // Resize frame to 224x224 exactly as the AI expects
    const resized = resize(frame, {
      scale: { width: 224, height: 224 },
      pixelFormat: 'rgb',
      dataType: 'uint8',
    });

    // Run inference synchronously
    const outputs = model.model.runSync([resized]);
    
    // In the next step, we will map these outputs back to labels.txt!
    console.log(outputs); 
  }, [model.state]);

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
        pixelFormat="rgb"
      />
      
      {/* Overlay UI will go here */}
      <View style={styles.overlay}>
        <Text style={styles.scanText}>Scanning for Cats...</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  centerText: {
    flex: 1,
    textAlign: 'center',
    textAlignVertical: 'center',
    fontSize: 18,
  },
  overlay: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
    borderRadius: 15,
  },
  scanText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  }
});