import React, { useEffect } from 'react';
import { StyleSheet, Text, View, ActivityIndicator } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, Frame } from 'react-native-vision-camera';

// @ts-ignore
import { useFrameProcessor } from 'react-native-vision-camera';

import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';

export default function CatScanner() {
  const device = useCameraDevice('back');
  const { hasPermission, requestPermission } = useCameraPermission();

  const model = useTensorflowModel(require('../../assets/models/model.tflite'), 'default' as any);
  
  const { resize } = useResizePlugin();

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  const frameProcessor = useFrameProcessor((frame: Frame) => {
    'worklet';
    if (model.state !== 'loaded') return;

    const resized = resize(frame, {
      scale: { width: 224, height: 224 },
      pixelFormat: 'rgb',
      dataType: 'uint8',
    });

    const outputs = model.model.runSync([resized as any]);
    console.log(outputs); 
  }, [model.state]);

  if (!hasPermission) return <Text style={styles.centerText}>Requesting Camera Permission...</Text>;
  if (device == null) return <Text style={styles.centerText}>No Camera Found</Text>;
  if (model.state === 'loading') return <ActivityIndicator style={styles.centerText} size="large" color="#00ff00" />;

  const cameraProps: any = {
    frameProcessor: frameProcessor
  };

  return (
    <View style={StyleSheet.absoluteFill}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        pixelFormat="rgb"
        {...cameraProps} 
      />
      
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