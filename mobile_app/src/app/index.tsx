import React from 'react';
import { StyleSheet, View } from 'react-native';
import CatScanner from '../components/CatScanner';

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <CatScanner />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});