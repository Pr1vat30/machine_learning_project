import { fileURLToPath, URL } from "node:url";
import vue from "@vitejs/plugin-vue";
import autoprefixer from "autoprefixer";
import tailwind from "tailwindcss";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  css: {
    postcss: {
      plugins: [tailwind(), autoprefixer()],
    },
  },
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  define: {
    __API_SERVER__: JSON.stringify("http://127.0.0.1:8080"),
  },
});
