export interface ScanRegion {
  /** Fractions of the visible preview, between 0 and 1. */
  left: number;
  top: number;
  width: number;
  height: number;
}

export function validateScanRegion(region: ScanRegion): void {
  if (![region.left, region.top, region.width, region.height].every(Number.isFinite)
    || region.left < 0 || region.top < 0 || region.width <= 0 || region.height <= 0
    || region.left + region.width > 1 || region.top + region.height > 1) {
    throw new RangeError('MRZ Reader: scanRegion must fit within the preview (0–1)');
  }
}

/** Match the preview's object-fit: cover crop without stretching the MRZ. */
export function captureGeometry(videoWidth: number, videoHeight: number, region: ScanRegion) {
  const previewWidth = 888;
  const previewHeight = 500;
  const scale = Math.max(previewWidth / videoWidth, previewHeight / videoHeight);
  const visibleWidth = previewWidth / scale;
  const visibleHeight = previewHeight / scale;
  return {
    sx: (videoWidth - visibleWidth) / 2 + visibleWidth * region.left,
    sy: (videoHeight - visibleHeight) / 2 + visibleHeight * region.top,
    sw: visibleWidth * region.width,
    sh: visibleHeight * region.height,
    width: Math.max(1, Math.round(previewWidth * region.width)),
    height: Math.max(1, Math.round(previewHeight * region.height)),
  };
}
