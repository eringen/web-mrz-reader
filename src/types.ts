export interface ValidationResult {
  documentNumber: boolean;
  dateOfBirth: boolean;
  expirationDate: boolean;
  composite: boolean;
  isValid: boolean;
}

export interface TD3ValidationResult {
  passportNumber: boolean;
  dateOfBirth: boolean;
  expirationDate: boolean;
  personalNumber: boolean;
  composite: boolean;
  isValid: boolean;
}

export interface TD1Result {
  "Document Type": string;
  Nationality: string;
  Surname: string;
  "Given Names": string;
  "Document Number": string;
  "Issuing Country": string;
  "Date of Birth": string;
  Gender: string;
  "Expiration Date": string;
  "Optional Data 1": string;
  "Optional Data 2": string;
  Validation: ValidationResult;
}

export interface TD2Result {
  "Document Type": string;
  Nationality: string;
  Surname: string;
  "Given Names": string;
  "Document Number": string;
  "Issuing Country": string;
  "Date of Birth": string;
  Gender: string;
  "Expiration Date": string;
  "Optional Data": string;
  Validation: ValidationResult;
}

export interface TD3Result {
  Nationality: string;
  Surname: string;
  "Given Names": string;
  "Passport Number": string;
  "Issuing Country": string;
  "Date of Birth": string;
  Gender: string;
  "Expiration Date": string;
  "Personal Number": string;
  Validation: TD3ValidationResult;
}

export type MRZResult = TD1Result | TD2Result | TD3Result;
