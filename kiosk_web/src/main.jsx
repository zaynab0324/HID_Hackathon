import React from "react";
import ReactDOM from "react-dom/client";
import { ThemeProvider } from "@aws-amplify/ui-react";
import "@aws-amplify/ui-react/styles.css";
import App from "./App.jsx";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif" }}>
          <h2>⚠️ The kiosk hit a display error</h2>
          <p>{String(this.state.error?.message || this.state.error)}</p>
          <button onClick={() => window.location.reload()} style={{ fontSize: 18, padding: "10px 20px" }}>
            Restart kiosk
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
import "./kiosk.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <ThemeProvider>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </ThemeProvider>
);
