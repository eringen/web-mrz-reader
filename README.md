# MRZ Reader

A browser-based MRZ (Machine Readable Zone) reader that uses webcam capture and OCR to extract document data. Supports passports, ID cards, and travel documents. All processing happens client-side for privacy.

## Doc - Story
https://eringen.com/blog/browser-based-passport-mrz-reader-with-tesseract-js

## Try it
https://eringen.com/workbench/web-mrz-reader/

## npm
https://www.npmjs.com/package/web-mrz-reader

## Features

- Real-time webcam capture
- Custom-trained Tesseract model optimized for MRZ recognition
- Supports TD1 (ID cards), TD2 (travel documents), and TD3 (passports)
- Check digit validation for all formats
- Structured data extraction (name, document number, dates, etc.)
- Visual bounding box feedback on recognized text
- Fully client-side processing (no data leaves the browser)

## Tech Stack

- **Vanilla JavaScript** - No frameworks, minimal tooling
- **Vite** - Dev server and production bundler
- **Tesseract.js v5** - JavaScript OCR engine with WebAssembly (installed via npm)
- **Custom MRZ Model** - Trained specifically for MRZ text recognition
- **Web APIs** - MediaDevices, Canvas, Blob

## Project Structure

```
web-mrz-reader/
├── index.html              # Main HTML page
├── index.js                # Core application logic (ES module)
├── vite.config.js          # Vite configuration
├── package.json            # Dependencies and scripts
├── public/
│   └── model/
│       └── mrz.traineddata.gz  # Custom Tesseract model for MRZ
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

## Usage

```bash
npm install
npm run dev
```

1. Open the local URL shown by Vite
2. Allow camera access when prompted
3. Position MRZ area within camera view
4. Click "Capture Image and Extract Text"
5. View extracted data in JSON format

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
