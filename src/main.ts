import Tesseract from 'tesseract.js';
import { isMRZ, extractMRZData, parseMrz } from './parsers.js';

export { isMRZ, extractMRZData, parseMrz };
export type { MRZResult, TD1Result, TD2Result, TD3Result, ValidationResult, TD3ValidationResult } from './types.js';

const constraints: MediaStreamConstraints = {
  video: {
    facingMode: { ideal: 'environment' },
    width: { ideal: 888 },
    height: { ideal: 500 },
  },
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
  capture: () => Promise<void>;
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
  video.muted = true;
  video.playsInline = true;
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
  let worker: Tesseract.Worker | null = null;
  let stopped = false;
  let captureId = 0;
  let activeCapture: Promise<void> | null = null;

  const workerPromise = Tesseract.createWorker('mrz', Tesseract.OEM.LSTM_ONLY, {
    workerPath,
    corePath,
    langPath,
  }).then(async (createdWorker) => {
    await createdWorker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
    });
    worker = createdWorker;
    return createdWorker;
  });
  void workerPromise.catch(() => undefined);

  const mediaDevices = navigator.mediaDevices;
  if (!mediaDevices?.getUserMedia) {
    options.onError?.('Error accessing the camera: MediaDevices API is unavailable');
  } else {
    mediaDevices.getUserMedia(constraints)
    .catch((err: unknown) => {
      const errorName = err && typeof err === 'object' && 'name' in err
        ? String(err.name)
        : '';
      if (errorName !== 'OverconstrainedError') throw err;
      return mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
    })
    .then((s) => {
      if (stopped) {
        s.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = s;
      video.srcObject = stream;
    })
    .catch((err) => {
      if (stopped) return;
      const message = err instanceof Error ? err.message : String(err);
      options.onError?.('Error accessing the camera: ' + message);
    });
  }

  function reset(): void {
    context.clearRect(0, 0, 888, 500);
  }

  function capture(): Promise<void> {
    if (stopped) return Promise.resolve();
    if (activeCapture) return activeCapture;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      options.onError?.('Camera is not ready yet');
      return Promise.resolve();
    }

    const currentCaptureId = ++captureId;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const capturePromise = new Promise<void>((resolve) => {
      try {
        canvas.toBlob((blob) => {
          if (!blob || stopped || currentCaptureId !== captureId) {
            resolve();
            return;
          }

          workerPromise
            .then((ocrWorker) => ocrWorker.recognize(blob, {}, {
              text: true,
              blocks: true,
              hocr: false,
              tsv: false,
            }))
            .then(({ data }) => {
              if (stopped || currentCaptureId !== captureId) return;
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
            })
            .catch((err: unknown) => {
              if (stopped || currentCaptureId !== captureId) return;
              const message = err instanceof Error ? err.message : String(err);
              options.onError?.('Error: ' + message);
              reset();
            })
            .finally(resolve);
        }, 'image/jpeg', 0.92);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        options.onError?.('Error capturing frame: ' + message);
        resolve();
      }
    });

    activeCapture = capturePromise;
    void capturePromise.finally(() => {
      if (activeCapture === capturePromise) {
        activeCapture = null;
      }
    });
    return capturePromise;
  }

  function terminateWorker(): void {
    if (worker) {
      const currentWorker = worker;
      worker = null;
      void currentWorker.terminate().catch(() => undefined);
      return;
    }

    void workerPromise.then((createdWorker) => {
      worker = null;
      return createdWorker.terminate();
    }).catch(() => undefined);
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
    if (stopped) return;
    stopped = true;
    captureId += 1;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    video.pause();
    video.srcObject = null;
    terminateWorker();
  }

  return { capture, reset, stop };
}
