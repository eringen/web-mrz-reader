import { initMRZReader } from './main.js';

const cbutton = document.getElementById('cbutton') as HTMLButtonElement;
const status = document.getElementById('scanStatus')!;
let scanning = true;
const mrzReader = initMRZReader({
  container: '#mrz-reader',
  workerPath: '/tesseract/worker.min.js',
  corePath: '/tesseract/',
  langPath: '/model/',
  autoScan: true,
  scanRegion: { left: 0.05, top: 0.55, width: 0.9, height: 0.35 },
  onResult: (result) => {
    if (!result) return;
    const mrzOutput = document.getElementById('mrzOutput');
    const output = document.getElementById('output');
    if (mrzOutput) mrzOutput.innerText = result.raw;
    if (output) output.innerText = JSON.stringify(result.parsed, null, 2);
    status.innerText = 'MRZ read successfully. Hold another document in the guide to scan it.';
  },
  onError: (error) => {
    const output = document.getElementById('output');
    if (output) output.innerText = error;
    status.innerText = 'Scanning stopped. Reload the page to retry.';
    scanning = false;
    cbutton.disabled = true;
  },
});

cbutton.addEventListener('click', () => {
  scanning = !scanning;
  if (scanning) {
    mrzReader.reset();
    mrzReader.startScanning();
    status.innerText = 'Fit all MRZ lines inside the green guide. Scanning automatically.';
  } else {
    mrzReader.pauseScanning();
    status.innerText = 'Scanning paused.';
  }
  cbutton.innerText = scanning ? 'Pause scanning' : 'Resume scanning';
});

window.addEventListener('pagehide', () => mrzReader.stop());
// A restored page needs a fresh reader after pagehide released the camera.
window.addEventListener('pageshow', (event) => { if (event.persisted) window.location.reload(); });
