import { WebView } from "react-native-webview";
import { SafeAreaView, StyleSheet } from "react-native";

export default function HomeScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <WebView
        source={{ uri: "https://children--bhavyasri2004.replit.app" }}
        startInLoadingState={true}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        style={styles.webview}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  webview: {
    flex: 1,
  },
});