import type { TD1Result, TD2Result, TD3Result, MRZResult } from './types.js';
import { validateTD1CheckDigits, validateTD2CheckDigits, validateTD3CheckDigits } from './checkdigit.js';

function parseGender(code: string): string {
  return code === 'M' ? 'Male' : code === 'F' ? 'Female' : 'Unspecified';
}

export function isMRZ(text: string): boolean {
  const mrzPattern = /[PIAC][A-Z<][A-Z]{3}[A-Z0-9<]+/;
  return mrzPattern.test(text);
}

export function splitIntoLines(mrzStr: string): string {
  if (mrzStr.length === 90) {
    return mrzStr.substr(0, 30) + '\n' + mrzStr.substr(30, 30) + '\n' + mrzStr.substr(60);
  } else if (mrzStr.length === 88) {
    return mrzStr.substr(0, 44) + '\n' + mrzStr.substr(44);
  } else if (mrzStr.length === 72) {
    return mrzStr.substr(0, 36) + '\n' + mrzStr.substr(36);
  } else {
    return 'Unrecognized MRZ length: ' + mrzStr.length;
  }
}

function parseTD1(mrz: string): TD1Result {
  const line1 = mrz.slice(0, 30);
  const line2 = mrz.slice(30, 60);
  const line3 = mrz.slice(60, 90);

  const docType = line1.substring(0, 2).replace(/</g, '').trim();
  const issuingCountry = line1.substring(2, 5).replace(/</g, '').trim();
  const documentNumber = line1.substring(5, 14).replace(/</g, '').trim();
  const optionalData1 = line1.substring(15, 30).replace(/</g, '').trim();

  const dateOfBirth = line2.substring(0, 6);
  const sex = line2[7];
  const expirationDate = line2.substring(8, 14);
  const nationality = line2.substring(15, 18).replace(/</g, '').trim();
  const optionalData2 = line2.substring(18, 29).replace(/</g, '').trim();

  const nameMatch = line3.match(/([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)/);
  let surname = '';
  let givenNames = '';
  if (nameMatch) {
    surname = nameMatch[1].replace(/</g, ' ').trim();
    givenNames = nameMatch[2].replace(/</g, ' ').trim();
  }

  const validation = validateTD1CheckDigits(line1, line2);

  return {
    "Document Type": docType,
    Nationality: nationality,
    Surname: surname,
    "Given Names": givenNames,
    "Document Number": documentNumber,
    "Issuing Country": issuingCountry,
    "Date of Birth": dateOfBirth,
    Gender: parseGender(sex),
    "Expiration Date": expirationDate,
    "Optional Data 1": optionalData1,
    "Optional Data 2": optionalData2,
    Validation: validation,
  };
}

function parseTD2(mrz: string): TD2Result {
  const firstLine = mrz.slice(0, 36);
  const secondLine = mrz.slice(36, 72);

  const docType = firstLine.substring(0, 2).replace(/</g, '').trim();
  const issuingCountry = firstLine.substring(2, 5).replace(/</g, '').trim();

  const nameField = firstLine.substring(5, 36);
  const nameMatch = nameField.match(/([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)/);
  let surname = '';
  let givenNames = '';
  if (nameMatch) {
    surname = nameMatch[1].replace(/</g, ' ').trim();
    givenNames = nameMatch[2].replace(/</g, ' ').trim();
  }

  const documentNumber = secondLine.substring(0, 9).replace(/</g, '').trim();
  const nationality = secondLine.substring(10, 13).replace(/</g, '').trim();
  const dateOfBirth = secondLine.substring(13, 19);
  const sex = secondLine[20];
  const expirationDate = secondLine.substring(21, 27);
  const optionalData = secondLine.substring(28, 35).replace(/</g, '').trim();

  const validation = validateTD2CheckDigits(secondLine);

  return {
    "Document Type": docType,
    Nationality: nationality,
    Surname: surname,
    "Given Names": givenNames,
    "Document Number": documentNumber,
    "Issuing Country": issuingCountry,
    "Date of Birth": dateOfBirth,
    Gender: parseGender(sex),
    "Expiration Date": expirationDate,
    "Optional Data": optionalData,
    Validation: validation,
  };
}

export function parseMrz(mrz: string): MRZResult | string {
  if (mrz.length === 90) {
    return parseTD1(mrz);
  } else if (mrz.length === 72) {
    return parseTD2(mrz);
  } else if (mrz.length === 88) {
    // TD3 (passport) parsing
    const firstLine = mrz.slice(0, 44);
    const firstLineMatch = firstLine.match(/P<([A-Z]{3})([A-Z<]*)<<([A-Z<]*)/);
    if (!firstLineMatch) {
      return "Unable to parse the first line of MRZ.";
    }

    const nationality = firstLineMatch[1];
    let surname = firstLineMatch[2].replace(/</g, ' ').trim();
    let givenNames = firstLineMatch[3].replace(/</g, ' ').trim();

    if (!givenNames) {
      const names = surname.split(' ');
      if (names.length > 1) {
        surname = names[0];
        givenNames = names.slice(1).join(' ');
      }
    }

    const secondLine = mrz.slice(44);
    const secondLineMatch = secondLine.match(/([A-Z0-9<]{9})([0-9])([A-Z]{3})([0-9]{6})([0-9])([MF<])([0-9]{6})([0-9])([A-Z0-9<]*)/);
    if (!secondLineMatch) {
      return "Unable to parse the second line of MRZ.";
    }

    const passportNumber = secondLineMatch[1].replace(/</g, '').trim();
    const issuingCountry = secondLineMatch[3];
    const dateOfBirth = secondLineMatch[4];
    const gender = secondLineMatch[6];
    const expirationDate = secondLineMatch[7];
    const personalNumber = secondLineMatch[9].replace(/</g, '').trim();
    const validation = validateTD3CheckDigits(secondLine);

    return {
      Nationality: nationality,
      Surname: surname,
      "Given Names": givenNames,
      "Passport Number": passportNumber,
      "Issuing Country": issuingCountry,
      "Date of Birth": dateOfBirth,
      Gender: parseGender(gender),
      "Expiration Date": expirationDate,
      "Personal Number": personalNumber,
      Validation: validation,
    } satisfies TD3Result;
  } else {
    return "Invalid MRZ length. Expected 88 (TD3), 72 (TD2), or 90 (TD1) characters, got " + mrz.length + ".";
  }
}

export function extractMRZData(text: string): string | undefined {
  const input = text.replace(/(\r\n|\n|\r)/gm, "");
  const regex = /[PIAC][A-Z<][A-Z]{3}[A-Z0-9<]+/;
  const match = input.match(regex);

  if (!match || !match[0]) {
    return;
  }

  let mrzRaw = match[0];
  let expectedLength: number;
  if (mrzRaw.length >= 90) {
    expectedLength = 90;
  } else if (mrzRaw.length >= 88) {
    expectedLength = 88;
  } else if (mrzRaw.length >= 72) {
    expectedLength = 72;
  } else {
    return;
  }
  mrzRaw = mrzRaw.substring(0, expectedLength);

  const parsedMrz = parseMrz(mrzRaw);

  if (typeof parsedMrz === 'string') {
    return parsedMrz;
  }

  const outputEl = document.getElementById('output');
  if (outputEl) {
    outputEl.innerText = JSON.stringify(parsedMrz, null, 2);
  }

  let result = splitIntoLines(mrzRaw);

  if ('Validation' in parsedMrz && !parsedMrz.Validation.isValid) {
    const failed: string[] = [];
    const validation = parsedMrz.Validation;
    if ('passportNumber' in validation && validation.passportNumber === false) failed.push('Passport Number');
    if ('documentNumber' in validation && validation.documentNumber === false) failed.push('Document Number');
    if (!validation.dateOfBirth) failed.push('Date of Birth');
    if (!validation.expirationDate) failed.push('Expiration Date');
    if ('personalNumber' in validation && validation.personalNumber === false) failed.push('Personal Number');
    if (!validation.composite) failed.push('Composite');
    result += '\n\nWarning: Check digit validation failed for: ' + failed.join(', ');
  }
  return result;
}
