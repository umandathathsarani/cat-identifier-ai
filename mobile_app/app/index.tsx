import { Redirect } from 'expo-router';

export default function Index() {
  // This tells Expo Router to look at your src component instead
    return <Redirect href={'/src/app/index' as any} />;
}