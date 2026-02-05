import Tesseract from 'tesseract.js';
import { isMRZ, extractMRZData } from './parsers.js';

const constraints: MediaStreamConstraints = {
  video: { facingMode: 'environment', width: { min: 888 }, height: { min: 500 } }
};

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('2d')!;

function reset(): void {
  const mrzOutput = document.getElementById('mrzOutput');
  const output = document.getElementById('output');
  if (mrzOutput) mrzOutput.innerText = '';
  if (output) output.innerText = '';
  context.clearRect(0, 0, 888, 500);
}

navigator.mediaDevices.getUserMedia(constraints)
  .then(function (stream) {
    const video = document.getElementById('camera') as HTMLVideoElement;
    video.srcObject = stream;
  })
  .catch(function (err) {
    console.error("Error accessing the camera: ", err);
  });

window.captureAndPerformOCR = captureAndPerformOCR;

function captureAndPerformOCR(): void {
  const cbutton = document.getElementById('cbutton') as HTMLButtonElement;
  cbutton.disabled = true;
  const video = document.getElementById('camera') as HTMLVideoElement;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  performOCR();
}

function performOCR(): void {
  canvas.toBlob((blob) => {
    if (!blob) return;
    Tesseract.recognize(blob, 'mrz', {
      workerPath: './tesseract/worker.min.js',
      corePath: './tesseract/',
      langPath: './model/',
    }).then(({ data }) => {
      const { text, words } = data;
      const mrzOutput = document.getElementById('mrzOutput');
      const cbutton = document.getElementById('cbutton') as HTMLButtonElement;
      if (isMRZ(text)) {
        if (mrzOutput) mrzOutput.innerText = "Detected MRZ: \n" + extractMRZData(text);
        drawBoundingBoxes(words);
        cbutton.disabled = false;
      } else {
        if (mrzOutput) mrzOutput.innerText = "";
        reset();
        cbutton.disabled = false;
      }
    }).catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      console.error(err);
      const output = document.getElementById('output');
      if (output) output.innerText = 'Error: ' + message;
      reset();
      const cbutton = document.getElementById('cbutton') as HTMLButtonElement;
      cbutton.disabled = false;
    });
  });
}

function drawBoundingBoxes(words: Tesseract.Word[]): void {
  context.strokeStyle = 'red';
  context.lineWidth = 2;

  words.forEach((word) => {
    const { bbox } = word;
    context.strokeRect(bbox.x0, bbox.y0, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0);
  });
}

declare global {
  interface Window {
    captureAndPerformOCR: () => void;
  }
}
