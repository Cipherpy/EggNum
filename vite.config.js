import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/EggNum/',
  plugins: [react()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
