import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'WebMRZ',
      formats: ['es'],
      fileName: 'web-mrz-reader',
    },
    rollupOptions: {
      external: ['tesseract.js'],
      output: {
        globals: {
          'tesseract.js': 'Tesseract',
        },
      },
    },
  },
});
