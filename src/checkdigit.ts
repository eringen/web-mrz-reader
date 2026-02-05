import type { ValidationResult, TD3ValidationResult } from './types.js';

export function calculateCheckDigit(input: string): number {
  const weights = [7, 3, 1];
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    let value: number;
    if (ch >= '0' && ch <= '9') {
      value = ch.charCodeAt(0) - '0'.charCodeAt(0);
    } else if (ch >= 'A' && ch <= 'Z') {
      value = ch.charCodeAt(0) - 'A'.charCodeAt(0) + 10;
    } else {
      value = 0; // '<' and any filler
    }
    sum += value * weights[i % 3];
  }
  return sum % 10;
}

export function validateTD1CheckDigits(line1: string, line2: string): ValidationResult {
  const documentNumber = calculateCheckDigit(line1.substring(5, 14)) === Number(line1[14]);
  const dateOfBirth = calculateCheckDigit(line2.substring(0, 6)) === Number(line2[6]);
  const expirationDate = calculateCheckDigit(line2.substring(8, 14)) === Number(line2[14]);
  const compositeData = line1.substring(5, 30) + line2.substring(0, 7) + line2.substring(8, 15) + line2.substring(18, 29);
  const composite = calculateCheckDigit(compositeData) === Number(line2[29]);

  return {
    documentNumber,
    dateOfBirth,
    expirationDate,
    composite,
    isValid: documentNumber && dateOfBirth && expirationDate && composite,
  };
}

export function validateTD2CheckDigits(secondLine: string): ValidationResult {
  const documentNumber = calculateCheckDigit(secondLine.substring(0, 9)) === Number(secondLine[9]);
  const dateOfBirth = calculateCheckDigit(secondLine.substring(13, 19)) === Number(secondLine[19]);
  const expirationDate = calculateCheckDigit(secondLine.substring(21, 27)) === Number(secondLine[27]);
  const compositeData = secondLine.substring(0, 10) + secondLine.substring(13, 20) + secondLine.substring(21, 28) + secondLine.substring(28, 35);
  const composite = calculateCheckDigit(compositeData) === Number(secondLine[35]);

  return {
    documentNumber,
    dateOfBirth,
    expirationDate,
    composite,
    isValid: documentNumber && dateOfBirth && expirationDate && composite,
  };
}

export function validateTD3CheckDigits(secondLine: string): TD3ValidationResult {
  const passportNumber = calculateCheckDigit(secondLine.substring(0, 9)) === Number(secondLine[9]);
  const dateOfBirth = calculateCheckDigit(secondLine.substring(13, 19)) === Number(secondLine[19]);
  const expirationDate = calculateCheckDigit(secondLine.substring(21, 27)) === Number(secondLine[27]);
  const personalNumber = calculateCheckDigit(secondLine.substring(28, 42)) === Number(secondLine[42]);
  const compositeData = secondLine.substring(0, 10) + secondLine.substring(13, 20) + secondLine.substring(21, 43);
  const composite = calculateCheckDigit(compositeData) === Number(secondLine[43]);

  return {
    passportNumber,
    dateOfBirth,
    expirationDate,
    personalNumber,
    composite,
    isValid: passportNumber && dateOfBirth && expirationDate && personalNumber && composite,
  };
}
