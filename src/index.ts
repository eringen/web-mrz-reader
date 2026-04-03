export { initMRZReader } from './main.js';
export { isMRZ, extractMRZData, parseMrz } from './parsers.js';
export { calculateCheckDigit, validateTD1CheckDigits, validateTD2CheckDigits, validateTD3CheckDigits } from './checkdigit.js';
export type { MRZResult, TD1Result, TD2Result, TD3Result, ValidationResult, TD3ValidationResult } from './types.js';
export type { MRZReaderOptions, MRZReaderInstance } from './main.js';
