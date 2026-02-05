const constraints = {
  video: { 'facingMode': 'environment', width: { min: 888 }, height: { min: 500 } }
};

const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

function reset() {
  document.getElementById('mrzOutput').innerText = '';
  document.getElementById('output').innerText = '';
  context.clearRect(0, 0, 888, 500);
}

navigator.mediaDevices.getUserMedia(constraints)
  .then(function (stream) {
    const video = document.getElementById('camera');
    video.srcObject = stream;
  })
  .catch(function (err) {
    console.error("Error accessing the camera: ", err);
  });

function captureAndPerformOCR() {
  document.getElementById('cbutton').disabled = true;
  const video = document.getElementById('camera');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  performOCR();
}

function performOCR() {
  canvas.toBlob((blob) => {
    Tesseract.recognize(blob, 'mrz', {
      langPath: './model/',
      corePath: './',
      workerPath: './worker.min.js',
      workerBlobURL: false
    }).then(({ data }) => {
      const { text, words } = data;
      if (isMRZ(text)) {
        document.getElementById('mrzOutput').innerText = "Detected MRZ: \n" + extractMRZData(text);
        drawBoundingBoxes(words);
        document.getElementById('cbutton').disabled = false;
      } else {
        document.getElementById('mrzOutput').innerText = "";
        reset();
        document.getElementById('cbutton').disabled = false;
      }
    }).catch((err) => {
      console.error(err);
      document.getElementById('output').innerText = 'Error: ' + (err && err.message ? err.message : err);
      reset();
      document.getElementById('cbutton').disabled = false;
    });
  });
}

function drawBoundingBoxes(words) {
  context.strokeStyle = 'red';
  context.lineWidth = 2;

  words.forEach((word) => {
    const { bbox } = word;
    context.strokeRect(bbox.x0, bbox.y0, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0);
  });
}

function isMRZ(text) {
  const mrzPattern = /[PIAC][A-Z<][A-Z]{3}[A-Z0-9<]+/;
  return mrzPattern.test(text);
}

function splitIntoLines(mrzStr) {
  if (mrzStr.length === 90) {
    return mrzStr.substr(0, 30) + '\n' + mrzStr.substr(30, 30) + '\n' + mrzStr.substr(60);
  } else if (mrzStr.length === 88) {
    return mrzStr.substr(0, 44) + '\n' + mrzStr.substr(44);
  } else if (mrzStr.length === 72) {
    return mrzStr.substr(0, 36) + '\n' + mrzStr.substr(36);
  } else {
    reset();
    return 'Unrecognized MRZ length: ' + mrzStr.length;
  }
}

function extractMRZData(text) {
  const input = text.replace(/(\r\n|\n|\r)/gm, "");
  const regex = /[PIAC][A-Z<][A-Z]{3}[A-Z0-9<]+/;
  const match = input.match(regex);

  if (!match || !match[0]) {
    return;
  }

  let mrzRaw = match[0];
  let expectedLength;
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
  document.getElementById('output').innerText = JSON.stringify(parsedMrz, null, 2);

  let result = splitIntoLines(mrzRaw);
  if (parsedMrz.Validation && !parsedMrz.Validation.isValid) {
    const failed = [];
    if (parsedMrz.Validation.passportNumber === false) failed.push('Passport Number');
    if (parsedMrz.Validation.documentNumber === false) failed.push('Document Number');
    if (!parsedMrz.Validation.dateOfBirth) failed.push('Date of Birth');
    if (!parsedMrz.Validation.expirationDate) failed.push('Expiration Date');
    if (parsedMrz.Validation.personalNumber === false) failed.push('Personal Number');
    if (!parsedMrz.Validation.composite) failed.push('Composite');
    result += '\n\nWarning: Check digit validation failed for: ' + failed.join(', ');
  }
  return result;
}

function calculateCheckDigit(input) {
  const weights = [7, 3, 1];
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    let value;
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

function validateTD1CheckDigits(line1, line2) {
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

function validateTD2CheckDigits(secondLine) {
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

function validateTD3CheckDigits(secondLine) {
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

function parseTD1(mrz) {
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
    Gender: sex === 'M' ? 'Male' : sex === 'F' ? 'Female' : 'Unspecified',
    "Expiration Date": expirationDate,
    "Optional Data 1": optionalData1,
    "Optional Data 2": optionalData2,
    Validation: validation,
  };
}

function parseTD2(mrz) {
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
    Gender: sex === 'M' ? 'Male' : sex === 'F' ? 'Female' : 'Unspecified',
    "Expiration Date": expirationDate,
    "Optional Data": optionalData,
    Validation: validation,
  };
}

function parseMrz(mrz) {
  if (mrz.length === 90) {
    return parseTD1(mrz);
  } else if (mrz.length === 72) {
    return parseTD2(mrz);
  } else if (mrz.length === 88) {
    // TD3 (passport) parsing
    const firstLine = mrz.slice(0, 44);
    const firstLineMatch = firstLine.match(/P<([A-Z]{3})([A-Z<]*)<<([A-Z<]*)/);
    if (!firstLineMatch) {
      reset();
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
      reset();
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
      Gender: gender === 'M' ? 'Male' : gender === 'F' ? 'Female' : 'Unspecified',
      "Expiration Date": expirationDate,
      "Personal Number": personalNumber,
      Validation: validation,
    };
  } else {
    reset();
    return "Invalid MRZ length. Expected 88 (TD3), 72 (TD2), or 90 (TD1) characters, got " + mrz.length + ".";
  }
}
