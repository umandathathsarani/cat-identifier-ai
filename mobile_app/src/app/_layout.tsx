import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack>
      {/* This hides the top header bar and shows only your scanner */}
      <Stack.Screen name="index" options={{ headerShown: false }} />
    </Stack>
  );
}