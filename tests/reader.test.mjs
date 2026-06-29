import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const validMRZ = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10';
const roi = { left: 0.05, top: 0.55, width: 0.9, height: 0.35 };
const tick = () => new Promise((resolve) => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

// Execute the real TS modules with browser/camera/worker boundaries controlled by each test.
function harness(config = {}) {
  const frames = new Map();
  const listeners = new Map();
  const elements = [];
  const jobs = [];
  let frameId = 0, creates = 0, terminates = 0, trackStops = 0, parameters = 0;
  let workerError;
  const stream = { getTracks: () => [{ stop: () => trackStops++ }] };
  const mediaRequests = [];
  function element(tag) {
    const el = {
      tag, style: {}, children: [], width: 300, height: 150,
      readyState: 2, videoWidth: 1280, videoHeight: 720, currentTime: 0,
      appendChild(child) { child.parent = this; this.children.push(child); },
      remove() { this.parent.children = this.parent.children.filter((child) => child !== this); },
      play: async () => {}, pause() {},
      context: { draws: [], boxes: [], clearRect() {},
        drawImage(...args) { if (config.drawFails) throw new Error('draw failed'); this.draws.push(args); },
        strokeRect(...args) { this.boxes.push(args); } },
      getContext() { return this.context; },
      toBlob(callback) {
        if (config.blobDeferred) config.blobDeferred.callback = callback;
        else callback(config.emptyBlob ? null : { width: this.width, height: this.height });
      },
    };
    if (tag === 'video' && !config.fallbackFrames) {
      el.requestVideoFrameCallback = (callback) => { frames.set(++frameId, callback); return frameId; };
      el.cancelVideoFrameCallback = (id) => frames.delete(id);
    }
    elements.push(el);
    return el;
  }
  const container = element('section');
  const document = {
    hidden: false, createElement: element, querySelector: () => container,
    addEventListener: (event, callback) => listeners.set(event, callback),
    removeEventListener: (event) => listeners.delete(event),
  };
  const worker = {
    async setParameters() { parameters++; if (config.parametersFail) throw new Error('parameters failed'); },
    recognize(blob, options, output) { const job = deferred(); jobs.push({ ...job, blob, options, output }); return job.promise; },
    async terminate() { terminates++; },
  };
  const Tesseract = {
    OEM: { LSTM_ONLY: 1 }, PSM: { SINGLE_BLOCK: '6' },
    async createWorker(_lang, _mode, options) { creates++; workerError = options.errorHandler; return config.workerDeferred ? config.workerDeferred.promise : worker; },
  };
  const sandbox = vm.createContext({
    document, navigator: { mediaDevices: { getUserMedia: (constraints) => {
      mediaRequests.push(constraints);
      if (config.media) return config.media(constraints, stream);
      return config.cameraDeferred ? config.cameraDeferred.promise : Promise.resolve(stream);
    } } },
    requestAnimationFrame: (callback) => { frames.set(++frameId, callback); return frameId; },
    cancelAnimationFrame: (id) => frames.delete(id),
  });
  const modules = new Map();
  function load(path) {
    if (modules.has(path)) return modules.get(path).exports;
    const module = { exports: {} };
    modules.set(path, module);
    const source = ts.transpileModule(readFileSync(path, 'utf8'), {
      compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
    }).outputText;
    const run = vm.runInContext(`(function(require, module, exports) { ${source}\n})`, sandbox);
    run((name) => name === 'tesseract.js' ? Tesseract : load(resolve(dirname(path), name.replace(/\.js$/, '.ts'))), module, module.exports);
    return module.exports;
  }
  const { initMRZReader } = load(resolve('src/main.ts'));
  const results = [], errors = [];
  const reader = initMRZReader({ container, onResult: (value) => results.push(value), onError: (value) => errors.push(value), ...config.options });
  return {
    reader, container, worker, stream, jobs, results, errors, frames, listeners, mediaRequests,
    elements, document, initMRZReader,
    video: elements.find((el) => el.tag === 'video'),
    snapshot: elements.find((el) => el.tag === 'canvas'),
    geometry: load(resolve('src/capture.ts')),
    stats: () => ({ creates, terminates, trackStops, parameters }),
    workerError: (error) => workerError(error),
    frame(time) {
      this.video.currentTime = time;
      const pending = [...frames.values()]; frames.clear(); pending.forEach((callback) => callback());
    },
    finish(index = jobs.length - 1, text = validMRZ) {
      jobs[index].resolve({ data: { text, words: [{ bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }] } });
    },
    hide(hidden) { document.hidden = hidden; listeners.get('visibilitychange')?.(); },
  };
}

test('reuses one worker, coalesces manual capture, keeps preview live', async () => {
  const h = harness(); await tick();
  const first = h.reader.capture();
  assert.equal(h.reader.capture(), first);
  await tick();
  assert.equal(h.jobs.length, 1);
  assert.equal(h.container.children[0].children.includes(h.snapshot), false);
  h.finish(); await first;
  const next = h.reader.capture(); await tick(); h.finish(); await next;
  assert.equal(h.stats().creates, 1);
  assert.equal(h.results.length, 2);
  h.reader.stop();
});

test('live scanning crops frames, requests text only, does not queue reads, deduplicates results', async () => {
  const h = harness({ options: { autoScan: true, scanRegion: roi } }); await tick();
  h.frame(1); await tick();
  assert.equal(h.jobs.length, 1);
  assert.equal(h.frames.size, 0);
  assert.equal(h.jobs[0].blob.width, 799);
  assert.equal(h.jobs[0].blob.height, 175);
  assert.equal(h.jobs[0].output.blocks, false);
  h.frame(2); await tick(); assert.equal(h.jobs.length, 1);
  h.finish(); await tick(); assert.equal(h.results.length, 1);
  h.frame(3); await tick(); h.finish(); await tick();
  assert.equal(h.results.length, 1);
  h.reader.reset(); h.frame(4); await tick(); h.finish(); await tick();
  assert.equal(h.results.length, 2);
  h.reader.stop();
});

test('invalid checks continue scanning without delivering an invalid live result', async () => {
  const h = harness({ options: { autoScan: true } }); await tick();
  h.frame(1); await tick(); h.finish(0, validMRZ.slice(0, -1) + '1'); await tick();
  assert.equal(h.results.length, 0);
  assert.equal(h.frames.size, 1);
  h.frame(2); await tick(); h.finish(); await tick(); assert.equal(h.results.length, 1);
  h.reader.stop();
});

test('stop settles capture even if terminated worker never settles recognition', async () => {
  const h = harness(); await tick(); const read = h.reader.capture(); await tick();
  let settled = false; read.then(() => { settled = true; });
  h.reader.stop(); h.reader.stop(); await tick();
  assert.equal(settled, true);
  assert.equal(h.stats().terminates, 1); assert.equal(h.stats().trackStops, 1);
  assert.equal(h.video.srcObject, null); assert.equal(h.container.children.length, 0);
  assert.equal(h.listeners.size, 0);
  h.finish(); await tick(); assert.equal(h.results.length, 0);
});

test('stop while loading worker never launches OCR and terminates late worker', async () => {
  const workerDeferred = deferred(); const h = harness({ workerDeferred }); await tick();
  const read = h.reader.capture(); h.reader.stop(); await read;
  workerDeferred.resolve(h.worker); await tick();
  assert.equal(h.jobs.length, 0); assert.equal(h.stats().terminates, 1);
  assert.equal(h.stats().parameters, 0);
});

test('stop during frame encoding prevents OCR', async () => {
  const blobDeferred = {}; const h = harness({ blobDeferred }); await tick();
  const read = h.reader.capture(); await tick(); h.reader.stop(); await read;
  blobDeferred.callback({}); await tick(); assert.equal(h.jobs.length, 0);
});

test('stop before permission resolves closes late camera stream', async () => {
  const cameraDeferred = deferred(); const h = harness({ cameraDeferred });
  h.reader.stop(); cameraDeferred.resolve(h.stream); await tick();
  assert.equal(h.stats().trackStops, 1); assert.equal(h.video.srcObject, null);
});

test('setup failures are reported without capture and clean up resources', async () => {
  const h = harness({ parametersFail: true }); await tick();
  assert.equal(h.errors.length, 1); assert.equal(h.stats().terminates, 1);
  assert.equal(h.stats().trackStops, 1); assert.equal(h.container.children.length, 0);
});

test('worker initialization error callback releases waiting capture even if createWorker hangs', async () => {
  const h = harness({ workerDeferred: deferred() }); await tick();
  const read = h.reader.capture(); let settled = false; read.then(() => { settled = true; });
  h.workerError('model could not load'); await tick();
  assert.equal(settled, true); assert.equal(h.errors.length, 1);
  assert.equal(h.stats().trackStops, 1); assert.equal(h.container.children.length, 0);
});

test('pause and visibility discard in-flight results; resume uses a fresh frame', async () => {
  const h = harness({ options: { autoScan: true } }); await tick();
  h.frame(1); await tick(); h.reader.pauseScanning(); h.finish(); await tick();
  assert.equal(h.results.length, 0); assert.equal(h.frames.size, 0);
  h.reader.startScanning(); h.frame(2); await tick(); h.hide(true); h.finish(); await tick();
  assert.equal(h.results.length, 0); assert.equal(h.frames.size, 0);
  h.hide(false); h.frame(3); await tick(); h.finish(); await tick();
  assert.equal(h.results.length, 1); h.reader.stop();
});

test('animation-frame fallback skips unchanged camera frames', async () => {
  const h = harness({ fallbackFrames: true, options: { autoScan: true } }); await tick();
  h.frame(1); await tick(); h.finish(); await tick();
  h.frame(1); await tick(); assert.equal(h.jobs.length, 1);
  h.frame(2); await tick(); assert.equal(h.jobs.length, 2); h.reader.stop();
});

test('reinitializing a container stops its previous reader without deleting host content', async () => {
  const h = harness(); await tick(); const content = { tag: 'p' }; h.container.appendChild(content);
  const second = h.initMRZReader({ container: h.container }); await tick();
  assert.equal(h.stats().terminates, 1); assert.equal(h.stats().trackStops, 1);
  assert.equal(h.container.children.length, 2); assert.ok(h.container.children.includes(content));
  second.stop();
});

test('camera fallback retries constraints only while reader is active', async () => {
  let attempts = 0;
  const h = harness({ media: async (_, stream) => {
    if (++attempts === 1) throw { name: 'OverconstrainedError' };
    return stream;
  } }); await tick(); assert.equal(attempts, 2); h.reader.stop();
  const cameraDeferred = deferred(); const stopped = harness({ cameraDeferred }); stopped.reader.stop();
  cameraDeferred.reject({ name: 'OverconstrainedError' }); await tick();
  assert.equal(stopped.mediaRequests.length, 1);
});

test('crop preserves aspect ratio and matches visible preview on portrait and landscape cameras', async () => {
  const h = harness(); await tick();
  for (const [width, height] of [[1920, 1080], [640, 480], [720, 1280]]) {
    const g = h.geometry.captureGeometry(width, height, roi);
    assert.ok(g.sx >= 0 && g.sy >= 0 && g.sx + g.sw <= width && g.sy + g.sh <= height);
    assert.ok(Math.abs(g.sw / g.sh - g.width / g.height) < 0.01);
  }
  assert.throws(() => h.geometry.validateScanRegion({ ...roi, width: 2 }));
  assert.throws(() => h.geometry.validateScanRegion({ ...roi, top: NaN }));
  assert.throws(() => h.geometry.validateScanRegion({ left: 0 }));
  h.reader.stop();
});

test('drawing and encoding failures settle captures and stop live retries', async () => {
  for (const config of [{ drawFails: true }, { emptyBlob: true }]) {
    const h = harness({ ...config, options: { autoScan: true } }); await tick();
    await h.reader.capture();
    assert.equal(h.errors.length, 1); assert.equal(h.frames.size, 0);
    assert.equal(h.jobs.length, 0); h.reader.stop();
  }
});

test('reset invalidates pending OCR without starting an overlapping read', async () => {
  const h = harness(); await tick();
  const read = h.reader.capture(); await tick(); h.reader.reset();
  assert.equal(h.reader.capture(), read);
  h.finish(); await read; assert.equal(h.results.length, 0);
  const fresh = h.reader.capture(); await tick(); h.finish(); await fresh;
  assert.equal(h.results.length, 1); h.reader.stop();
});

test('manual cropped capture maps boxes back into the preview', async () => {
  const h = harness({ options: { scanRegion: roi } }); await tick();
  const read = h.reader.capture(); await tick(); h.finish(); await read;
  assert.equal(h.jobs[0].output.blocks, true);
  const overlay = h.elements.filter((el) => el.tag === 'canvas')[1];
  const [x, y, width, height] = overlay.context.boxes[0];
  assert.equal(x, 888 * roi.left); assert.equal(y, 500 * roi.top);
  assert.ok(Math.abs(width - 10) < 0.01); assert.equal(height, 10);
  h.reader.stop();
});

test('recognition rejection settles capture and allows a manual retry', async () => {
  const h = harness(); await tick(); const first = h.reader.capture(); await tick();
  h.jobs[0].reject(new Error('OCR failed')); await first;
  assert.equal(h.errors.length, 1);
  const next = h.reader.capture(); await tick(); h.finish(); await next;
  assert.equal(h.results.length, 1); h.reader.stop();
});
