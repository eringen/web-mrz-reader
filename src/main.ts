import Tesseract from 'tesseract.js';
import { isMRZ, extractMRZData, parseMrz } from './parsers.js';

export { isMRZ, extractMRZData, parseMrz };
export type { MRZResult, TD1Result, TD2Result, TD3Result, ValidationResult, TD3ValidationResult } from './types.js';

const constraints: MediaStreamConstraints = {
  video: { facingMode: 'environment', width: { min: 888 }, height: { min: 500 } }
};

export interface MRZReaderOptions {
  container: string | HTMLElement;
  workerPath?: string;
  corePath?: string;
  langPath?: string;
  onResult?: (result: ReturnType<typeof extractMRZData>) => void;
  onError?: (error: string) => void;
}

export interface MRZReaderInstance {
  capture: () => void;
  reset: () => void;
  stop: () => void;
}

export function initMRZReader(options: MRZReaderOptions): MRZReaderInstance {
  const container = typeof options.container === 'string'
    ? document.querySelector(options.container)
    : options.container;

  if (!container) {
    throw new Error('MRZ Reader: container not found');
  }

  const workerPath = options.workerPath ?? '/tesseract/worker.min.js';
  const corePath = options.corePath ?? '/tesseract/';
  const langPath = options.langPath ?? '/model/';

  const video = document.createElement('video');
  video.autoplay = true;
  video.width = 888;
  video.height = 500;
  video.style.width = '100%';
  video.style.height = '100%';
  video.style.objectFit = 'cover';

  const canvas = document.createElement('canvas');
  canvas.width = 888;
  canvas.height = 500;
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';

  const context = canvas.getContext('2d')!;

  const wrap = document.createElement('div');
  wrap.style.position = 'relative';
  wrap.style.width = '100%';
  wrap.style.maxWidth = '888px';
  wrap.style.aspectRatio = '888 / 500';
  wrap.style.background = '#111';
  wrap.style.borderRadius = '8px';
  wrap.style.overflow = 'hidden';
  wrap.style.border = '1px solid #222';
  wrap.appendChild(video);
  wrap.appendChild(canvas);

  if (typeof options.container === 'string') {
    container.innerHTML = '';
    container.appendChild(wrap);
  } else {
    container.appendChild(wrap);
  }

  let stream: MediaStream | null = null;

  navigator.mediaDevices.getUserMedia(constraints)
    .then((s) => {
      stream = s;
      video.srcObject = stream;
    })
    .catch((err) => {
      const message = err instanceof Error ? err.message : String(err);
      options.onError?.('Error accessing the camera: ' + message);
    });

  function reset(): void {
    context.clearRect(0, 0, 888, 500);
  }

  function capture(): void {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    performOCR();
  }

  function performOCR(): void {
    canvas.toBlob((blob) => {
      if (!blob) return;
      Tesseract.recognize(blob, 'mrz', {
        workerPath,
        corePath,
        langPath,
      }).then(({ data }) => {
        const { text, words } = data;
        if (isMRZ(text)) {
          const result = extractMRZData(text);
          if (result) {
            options.onResult?.(result);
          }
          drawBoundingBoxes(words);
        } else {
          reset();
        }
      }).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        options.onError?.('Error: ' + message);
        reset();
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

  function stop(): void {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
  }

  return { capture, reset, stop };
}
