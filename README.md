# MRZ Reader

A browser-based MRZ (Machine Readable Zone) reader that uses webcam capture and OCR to extract document data. Supports passports, ID cards, and travel documents. All processing happens client-side for privacy.

## Doc - Story
https://eringen.com/blog/browser-based-passport-mrz-reader-with-tesseract-js

## Try it
https://eringen.com/workbench/web-mrz-reader/

## npm
https://www.npmjs.com/package/web-mrz-reader

## React Wrapper
https://www.npmjs.com/package/web-mrz-reader-react

For React projects, use the official wrapper:
```bash
npm install web-mrz-reader-react
```
See [web-mrz-reader-react](https://github.com/eringen/web-mrz-reader-react) for docs.

## NPM User Guide

### 1. Install

```bash
npm install web-mrz-reader tesseract.js
```

### 2. Copy Static Assets

Copy the trained model and Tesseract runtime files into your project's public directory:

```bash
mkdir -p public/model public/tesseract

# MRZ-trained OCR model
cp node_modules/web-mrz-reader/public/model/mrz.traineddata.gz public/model/

# Tesseract.js worker and WASM cores
cp node_modules/web-mrz-reader/public/tesseract/* public/tesseract/
```

### 3. Add HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MRZ Reader</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
    h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 1.5rem; }
    #mrz-reader { width: 100%; max-width: 888px; }
    button { margin-top: 1.25rem; padding: 0.75rem 1.5rem; font-size: 0.95rem; font-weight: 500; color: #fff; background: #1a73e8; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1557b0; }
    button:disabled { background: #333; color: #666; cursor: not-allowed; }
    #mrzOutput { margin-top: 1.25rem; font-family: monospace; font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; color: #4ade80; max-width: 888px; width: 100%; }
    #output { margin-top: 0.75rem; font-family: monospace; font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap; color: #94a3b8; max-width: 888px; width: 100%; }
  </style>
</head>
<body>
  <h1>MRZ Reader</h1>
  <div id="mrz-reader"></div>
  <button id="cbutton">Capture &amp; Read MRZ</button>
  <p id="mrzOutput"></p>
  <p id="output"></p>
  <script type="module" src="./main.js"></script>
</body>
</html>
```

### 4. Add JavaScript

```js
import { initMRZReader } from 'web-mrz-reader';

const mrzReader = initMRZReader({
  container: '#mrz-reader',
  workerPath: '/tesseract/worker.min.js',
  corePath: '/tesseract/',
  langPath: '/model/',
  onResult: (result) => {
    if (!result) return;
    document.getElementById('mrzOutput').innerText = result.raw;
    document.getElementById('output').innerText = JSON.stringify(result.parsed, null, 2);
  },
  onError: (error) => {
    document.getElementById('output').innerText = error;
  },
});

document.getElementById('cbutton').addEventListener('click', () => {
  const btn = document.getElementById('cbutton');
  btn.disabled = true;
  mrzReader.capture();
  setTimeout(() => { btn.disabled = false; }, 3000);
});
```

### 5. Serve with a dev server

Using Vite:

```bash
npm install -D vite
npx vite
```

Or any static server that supports ES modules. Open the URL, allow camera access, and click the capture button.

### API Reference

```js
const mrzReader = initMRZReader({
  container: '#mrz-reader',       // CSS selector or HTMLElement (required)
  workerPath: '/tesseract/worker.min.js',  // path to tesseract worker
  corePath: '/tesseract/',        // path to tesseract WASM cores
  langPath: '/model/',            // path to mrz.traineddata.gz
  onResult: (result) => {},       // called when MRZ is detected
  onError: (error) => {},         // called on camera or OCR errors
});

mrzReader.capture();   // capture frame and run OCR
mrzReader.reset();     // clear canvas
mrzReader.stop();      // stop camera stream
```

### Standalone Parsing (no camera)

If you already have MRZ text and just want to parse it:

```js
import { isMRZ, extractMRZData, parseMrz } from 'web-mrz-reader';

const text = 'P<GBRNEGUS<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n1234567890GBR8001015M3001015<<<<<<<<<<<<<<00';

if (isMRZ(text)) {
  const result = extractMRZData(text);
  console.log(result);
}

// Or parse directly if you know the format
const parsed = parseMrz(text);
console.log(parsed);
```

---------------------

## Features

- Real-time webcam capture
- Custom-trained Tesseract model optimized for MRZ recognition
- Supports TD1 (ID cards), TD2 (travel documents), and TD3 (passports)
- Check digit validation for all formats
- Structured data extraction (name, document number, dates, etc.)
- Visual bounding box feedback on recognized text
- Fully client-side processing (no data leaves the browser)

## Tech Stack

- **TypeScript** - Strict mode, modular architecture
- **Vite** - Dev server and production bundler (handles TS natively)
- **Tesseract.js v5** - JavaScript OCR engine with WebAssembly (installed via npm)
- **Custom MRZ Model** - Trained specifically for MRZ text recognition
- **Web APIs** - MediaDevices, Canvas, Blob

## Project Structure

```
web-mrz-reader/
├── index.html              # Main HTML page
├── src/
│   ├── index.ts            # Package entry point
│   ├── main.ts             # initMRZReader function
│   ├── main.demo.ts        # Demo app entry point
│   ├── types.ts            # Interfaces: TD1/TD2/TD3 results, validation
│   ├── checkdigit.ts       # Check digit calculation and validation
│   └── parsers.ts          # MRZ parsing, extraction, format detection
├── tsconfig.json           # TypeScript configuration (strict mode)
├── vite.config.ts          # Vite library build config
├── package.json            # Dependencies and scripts
├── public/
│   ├── model/
│   │   └── mrz.traineddata.gz  # Custom Tesseract model for MRZ
│   └── tesseract/              # Tesseract.js runtime files (auto-copied)
└── model_training.md       # Guide for training an improved model
```

## Supported Formats

| Format | Document Type    | Structure              |
|--------|------------------|------------------------|
| TD1    | ID cards         | 3 lines x 30 chars     |
| TD2    | Travel documents | 2 lines x 36 chars     |
| TD3    | Passports        | 2 lines x 44 chars     |

## Extracted Data Fields

**All formats:** Nationality, Surname, Given Names, Document Number, Issuing Country, Date of Birth, Gender, Expiration Date, Validation (check digits)

**TD1 additionally:** Document Type, Optional Data 1, Optional Data 2

**TD2 additionally:** Document Type, Optional Data

**TD3 additionally:** Passport Number, Personal Number

## Local Development

```bash
npm install
npm run dev
```

1. Open the local URL shown by Vite
2. Allow camera access when prompted
3. Position MRZ area within camera view
4. Click "Capture & Read MRZ"
5. View extracted data in JSON format

### Type Check

```bash
npm run typecheck
```

### Production Build

```bash
npm run build
```

The output in `dist/` can be deployed to any static host.

## Requirements

- Node.js (for build tooling)
- Modern browser with WebAssembly support
- Camera access permission
- HTTPS or localhost (required for camera API)

## License

Tesseract.js is licensed under Apache-2.0.
