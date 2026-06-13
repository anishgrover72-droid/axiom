import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /reason to the FastAPI backend during development.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/reason": "http://localhost:8000", "/health": "http://localhost:8000" },
  },
});
