import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = __dirname;

function findModulePath(moduleName) {
  let current = packageRoot;
  while (true) {
    const candidate = path.join(current, 'node_modules', moduleName);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

const tesseractJsPath = findModulePath('tesseract.js');
const tesseractCorePath = findModulePath('tesseract.js-core');

if (!tesseractJsPath) {
  console.warn('Warning: tesseract.js not found, skipping tesseract file copy');
  process.exit(0);
}

const dest = path.join(packageRoot, 'public', 'tesseract');
fs.mkdirSync(dest, { recursive: true });

const files = [
  [path.join(tesseractJsPath, 'dist', 'worker.min.js'), 'worker.min.js'],
];

if (tesseractCorePath) {
  files.push([path.join(tesseractCorePath, 'tesseract-core-simd-lstm.wasm.js'), 'tesseract-core-simd-lstm.wasm.js']);
  files.push([path.join(tesseractCorePath, 'tesseract-core-simd.wasm.js'), 'tesseract-core-simd.wasm.js']);
  files.push([path.join(tesseractCorePath, 'tesseract-core-lstm.wasm.js'), 'tesseract-core-lstm.wasm.js']);
  files.push([path.join(tesseractCorePath, 'tesseract-core.wasm.js'), 'tesseract-core.wasm.js']);
}

for (const [src, name] of files) {
  if (fs.existsSync(src)) {
    const destPath = path.join(dest, name);
    fs.copyFileSync(src, destPath);
    console.log(`Copied ${name}`);
  } else {
    console.warn(`Warning: ${src} not found, skipping`);
  }
}
