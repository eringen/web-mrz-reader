import { initMRZReader } from './main.js';

const mrzReader = initMRZReader({
  container: '#mrz-reader',
  workerPath: '/tesseract/worker.min.js',
  corePath: '/tesseract/',
  langPath: '/model/',
  onResult: (result) => {
    if (!result) return;
    const mrzOutput = document.getElementById('mrzOutput');
    const output = document.getElementById('output');
    if (mrzOutput) mrzOutput.innerText = result.raw;
    if (output) output.innerText = JSON.stringify(result.parsed, null, 2);
  },
  onError: (error) => {
    const output = document.getElementById('output');
    if (output) output.innerText = error;
  },
});

const cbutton = document.getElementById('cbutton') as HTMLButtonElement;
cbutton.addEventListener('click', () => {
  cbutton.disabled = true;
  mrzReader.capture();
  setTimeout(() => { cbutton.disabled = false; }, 3000);
});
