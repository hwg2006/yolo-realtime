// Vite 构建配置：启用 Vue 单文件组件，使用相对资源路径以适配 Electron 的 file:// 加载。
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  // 相对路径，便于 Electron 以 file:// 加载打包产物
  base: './',
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
})
