import Tesseract from 'tesseract.js';
import { isMRZ, extractMRZData, parseMrz } from './parsers.js';
import { captureGeometry, validateScanRegion, type ScanRegion } from './capture.js';

export { isMRZ, extractMRZData, parseMrz };
export type { ScanRegion } from './capture.js';
export type { MRZResult, TD1Result, TD2Result, TD3Result, ValidationResult, TD3ValidationResult } from './types.js';

const instances = new WeakMap<Element, MRZReaderInstance>();
const constraints: MediaStreamConstraints = {
  video: {
    facingMode: { ideal: 'environment' },
    width: { ideal: 888 },
    height: { ideal: 500 },
  },
  audio: false,
};

export interface MRZReaderOptions {
  container: string | HTMLElement;
  workerPath?: string;
  corePath?: string;
  langPath?: string;
  /** Start continuous scanning once the camera and OCR worker are ready. */
  autoScan?: boolean;
  /** Visible preview area to scan. Defaults to the entire preview. */
  scanRegion?: ScanRegion;
  /** Defaults to false during continuous scanning, true for manual capture. */
  drawBoundingBoxes?: boolean;
  onResult?: (result: ReturnType<typeof extractMRZData>) => void;
  onError?: (error: string) => void;
}

export interface MRZReaderInstance {
  capture: () => Promise<void>;
  startScanning: () => void;
  pauseScanning: () => void;
  reset: () => void;
  /** Final cleanup; initialize a new reader to restart the camera. */
  stop: () => void;
}

export function initMRZReader(options: MRZReaderOptions): MRZReaderInstance {
  const container = typeof options.container === 'string'
    ? document.querySelector(options.container) : options.container;
  if (!container) throw new Error('MRZ Reader: container not found');

  const region = { ...(options.scanRegion ?? { left: 0, top: 0, width: 1, height: 1 }) };
  validateScanRegion(region);
  const video = document.createElement('video');
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.width = 888;
  video.height = 500;
  video.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block';

  // Keep the camera preview live: the OCR snapshot is never displayed over it.
  const snapshot = document.createElement('canvas');
  const canvas = document.createElement('canvas');
  canvas.width = 888;
  canvas.height = 500;
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none';
  const context = canvas.getContext('2d');
  const captureContext = snapshot.getContext('2d');
  if (!context || !captureContext) throw new Error('MRZ Reader: canvas is unavailable');
  const overlay = context;
  const frameContext = captureContext;

  const wrap = document.createElement('div');
  wrap.style.cssText = 'position:relative;width:100%;max-width:888px;aspect-ratio:888/500;background:#111;border-radius:8px;overflow:hidden;border:1px solid #222';
  wrap.appendChild(video);
  wrap.appendChild(canvas);
  if (options.scanRegion) {
    const guide = document.createElement('div');
    guide.style.cssText = `position:absolute;pointer-events:none;box-sizing:border-box;border:2px dashed #4ade80;left:${region.left * 100}%;top:${region.top * 100}%;width:${region.width * 100}%;height:${region.height * 100}%`;
    wrap.appendChild(guide);
  }
  instances.get(container)?.stop();
  container.appendChild(wrap);

  let stream: MediaStream | null = null;
  let worker: Tesseract.Worker | null = null;
  let stopped = false;
  let scanning = options.autoScan ?? false;
  let captureId = 0;
  let activeCapture: Promise<void> | null = null;
  let scheduledFrame: number | null = null;
  let usesVideoCallback = false;
  let lastFrameTime = -1;
  let lastResult = '';
  let cancelCapture: (() => void) | null = null;

  function reportError(prefix: string, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    options.onError?.(`${prefix}: ${message}`);
  }

  function terminateWorker(): void {
    const current = worker;
    worker = null;
    if (current) void current.terminate().catch(() => undefined);
  }

  const workerPromise = (async () => {
    const created = await Tesseract.createWorker('mrz', Tesseract.OEM.LSTM_ONLY, {
      workerPath: options.workerPath ?? '/tesseract/worker.min.js',
      corePath: options.corePath ?? '/tesseract/',
      langPath: options.langPath ?? '/model/',
      // Some Tesseract v5 initialization errors never settle createWorker().
      errorHandler: (error: unknown) => {
        if (!stopped) {
          stop();
          reportError('OCR worker error', error);
        }
      },
    });
    worker = created;
    if (stopped) {
      terminateWorker();
      return null;
    }
    await created.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<',
    });
    return stopped ? null : created;
  })().catch((error: unknown) => {
    if (!stopped) {
      stop();
      reportError('Error initializing OCR', error);
    }
    return null;
  });
  void workerPromise.catch(() => undefined);

  function ready(): boolean {
    return !!stream && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  }

  function cancelScheduledFrame(): void {
    if (scheduledFrame === null) return;
    if (usesVideoCallback) video.cancelVideoFrameCallback(scheduledFrame);
    else cancelAnimationFrame(scheduledFrame);
    scheduledFrame = null;
  }

  function scheduleFrame(): void {
    if (stopped || !scanning || document.hidden || activeCapture || scheduledFrame !== null) return;
    const onFrame = () => {
      scheduledFrame = null;
      if (stopped || !scanning || document.hidden) return;
      if (!ready() || video.currentTime === lastFrameTime) {
        scheduleFrame();
        return;
      }
      lastFrameTime = video.currentTime;
      void capture().catch(() => pauseScanning());
    };
    usesVideoCallback = typeof video.requestVideoFrameCallback === 'function';
    scheduledFrame = usesVideoCallback
      ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
  }

  function visibilityChanged(): void {
    if (document.hidden) {
      cancelScheduledFrame();
      captureId += 1;
    } else scheduleFrame();
  }

  function reset(): void {
    captureId += 1;
    lastResult = '';
    overlay.clearRect(0, 0, canvas.width, canvas.height);
  }

  function startScanning(): void {
    if (stopped) return;
    scanning = true;
    scheduleFrame();
  }

  function pauseScanning(): void {
    scanning = false;
    captureId += 1;
    cancelScheduledFrame();
  }

  async function readFrame(id: number, live: boolean): Promise<void> {
    try {
      const showBoxes = options.drawBoundingBoxes ?? !live;
      const ocrWorker = await workerPromise;
      if (!ocrWorker || stopped || id !== captureId || !ready()) return;
      const geometry = captureGeometry(video.videoWidth, video.videoHeight, region);
      if (snapshot.width !== geometry.width) snapshot.width = geometry.width;
      if (snapshot.height !== geometry.height) snapshot.height = geometry.height;
      frameContext.drawImage(video, geometry.sx, geometry.sy, geometry.sw, geometry.sh,
        0, 0, snapshot.width, snapshot.height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        snapshot.toBlob((value) => value ? resolve(value) : reject(new Error('Empty camera frame')),
          'image/jpeg', 0.92);
      });
      if (stopped || id !== captureId) return;
      const { data } = await ocrWorker.recognize(blob, {}, {
        text: true, blocks: showBoxes, hocr: false, tsv: false,
      });
      if (stopped || id !== captureId || (live && document.hidden)) return;
      overlay.clearRect(0, 0, canvas.width, canvas.height);
      const result = extractMRZData(data.text);
      // Live OCR keeps trying until all checks pass; manual captures retain diagnostics.
      if (!result || (live && (typeof result.parsed === 'string' || !result.parsed.Validation.isValid))) return;
      if (showBoxes) {
        overlay.strokeStyle = 'red';
        overlay.lineWidth = 2;
        const scaleX = canvas.width * region.width / snapshot.width;
        const scaleY = canvas.height * region.height / snapshot.height;
        for (const { bbox } of data.words ?? []) {
          overlay.strokeRect(canvas.width * region.left + bbox.x0 * scaleX,
            canvas.height * region.top + bbox.y0 * scaleY,
            (bbox.x1 - bbox.x0) * scaleX, (bbox.y1 - bbox.y0) * scaleY);
        }
      }
      if (!live || result.raw !== lastResult) {
        lastResult = result.raw;
        options.onResult?.(result);
      }
    } catch (error: unknown) {
      if (stopped || id !== captureId) return;
      pauseScanning();
      overlay.clearRect(0, 0, canvas.width, canvas.height);
      reportError('Error reading MRZ', error);
    }
  }

  function capture(): Promise<void> {
    if (stopped) return Promise.resolve();
    if (activeCapture) return activeCapture;
    if (!ready()) {
      if (!scanning) options.onError?.('Camera is not ready yet');
      scheduleFrame();
      return Promise.resolve();
    }
    cancelScheduledFrame();
    const cancelled = new Promise<void>((resolve) => { cancelCapture = resolve; });
    const job = Promise.race([readFrame(++captureId, scanning), cancelled]);
    const completion = job.finally(() => {
      activeCapture = null;
      cancelCapture = null;
      scheduleFrame();
    });
    activeCapture = completion;
    return completion;
  }

  function stop(): void {
    if (stopped) return;
    stopped = true;
    pauseScanning();
    cancelCapture?.(); // Worker termination does not settle outstanding Tesseract v5 jobs.
    terminateWorker();
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
    document.removeEventListener('visibilitychange', visibilityChanged);
    reset();
    snapshot.width = snapshot.height = 0;
    wrap.remove();
    if (instances.get(container!) === instance) instances.delete(container!);
  }

  const instance = { capture, startScanning, pauseScanning, reset, stop };
  instances.set(container, instance);
  document.addEventListener('visibilitychange', visibilityChanged);

  void (async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('MediaDevices API is unavailable');
      let acquired: MediaStream;
      try {
        acquired = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (error) {
        if (stopped) return;
        if (!error || typeof error !== 'object' || !('name' in error) || error.name !== 'OverconstrainedError') throw error;
        acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
      }
      if (stopped) {
        acquired.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = acquired;
      video.srcObject = stream;
      await video.play();
      await workerPromise;
      scheduleFrame();
    } catch (error: unknown) {
      if (!stopped) {
        stop();
        reportError('Error accessing the camera', error);
      }
    }
  })();

  return instance;
}
