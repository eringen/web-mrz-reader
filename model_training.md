# Training an Improved Tesseract MRZ Model

This guide walks through creating an improved Tesseract LSTM model for Machine Readable Zone (MRZ) recognition, starting from the existing `mrz.traineddata` model.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Understanding the Current Model](#2-understanding-the-current-model)
3. [Setting Up the Training Environment](#3-setting-up-the-training-environment)
4. [Preparing Training Data](#4-preparing-training-data)
5. [Image Augmentation](#5-image-augmentation)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluating the Model](#7-evaluating-the-model)
8. [Packaging for Web](#8-packaging-for-web)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

- macOS, Linux, or WSL on Windows
- Python 3.8+
- Tesseract 5.x with training tools
- OCR-B font installed on your system
- At least 4 GB of free disk space for training artifacts

---

## 2. Understanding the Current Model

The existing `model/mrz.traineddata.gz` is a compressed Tesseract LSTM model trained specifically for MRZ text. MRZ uses a restricted character set:

| Characters | Count | Examples         |
|------------|-------|------------------|
| Letters    | 26    | A-Z (uppercase)  |
| Digits     | 10    | 0-9              |
| Filler     | 1     | < (angle bracket) |

MRZ text appears in the **OCR-B** monospaced font, printed on identity documents in one of three formats:

- **TD1** (ID cards): 3 lines x 30 characters = 90 characters
- **TD2** (travel documents): 2 lines x 36 characters = 72 characters
- **TD3** (passports): 2 lines x 44 characters = 88 characters

The model's job is to recognize these characters from camera-captured images, which may include blur, noise, skew, uneven lighting, and reflections.

---

## 3. Setting Up the Training Environment

### 3.1 Install Tesseract with Training Tools

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr libtesseract-dev
sudo apt install tesseract-ocr-tools   # training tools

# Verify installation
tesseract --version
text2image --version
lstmtraining --version
combine_tessdata --version
```

All four commands must be available. If `text2image` or `lstmtraining` are missing, the training tools were not installed.

### 3.2 Install Python Dependencies

```bash
python3 -m venv mrz-training-env
source mrz-training-env/bin/activate

pip install opencv-python numpy pillow tqdm
```

### 3.3 Install the OCR-B Font

The OCR-B font is the standard for MRZ printing. Download it from a font provider and install it:

```bash
# macOS - copy to system fonts
cp OCR-B.ttf /Library/Fonts/

# Linux - copy to user fonts
mkdir -p ~/.fonts
cp OCR-B.ttf ~/.fonts/
fc-cache -fv

# Verify Tesseract can see it
text2image --list_available_fonts --fonts_dir=/Library/Fonts | grep -i ocr
```

If you cannot obtain OCR-B, these monospaced alternatives work as supplementary training fonts:
- **Courier New**
- **DejaVu Sans Mono**
- **Liberation Mono**

### 3.4 Create the Working Directory

```bash
mkdir -p mrz-training/{ground-truth,augmented,output,eval}
cd mrz-training
```

---

## 4. Preparing Training Data

### 4.1 Generate MRZ Sample Text

Create a Python script to generate randomized but structurally valid MRZ strings.

**File: `generate_training_text.py`**

```python
import random
import string

LETTERS = string.ascii_uppercase
DIGITS = string.digits
MRZ_CHARS = LETTERS + DIGITS + "<"

def random_alpha(n):
    return "".join(random.choices(LETTERS, k=n))

def random_digit(n):
    return "".join(random.choices(DIGITS, k=n))

def random_mrz_char(n):
    return "".join(random.choices(MRZ_CHARS, k=n))

def pad(s, length):
    return s.ljust(length, "<")[:length]

def calculate_check_digit(data):
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            value = int(ch)
        elif ch.isalpha():
            value = ord(ch) - ord("A") + 10
        else:
            value = 0
        total += value * weights[i % 3]
    return str(total % 10)

def generate_name():
    surname = random_alpha(random.randint(3, 12))
    given1 = random_alpha(random.randint(3, 10))
    given2 = random_alpha(random.randint(3, 8)) if random.random() > 0.3 else ""
    if given2:
        return f"{surname}<<{given1}<{given2}"
    return f"{surname}<<{given1}"

def generate_date():
    y = random.randint(50, 99) if random.random() > 0.5 else random.randint(0, 30)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:02d}{m:02d}{d:02d}"

def generate_td3():
    country = random_alpha(3)
    name = pad(generate_name(), 39)
    line1 = f"P<{country}{name}"

    doc_num = random_mrz_char(9)
    doc_check = calculate_check_digit(doc_num)
    nationality = random_alpha(3)
    dob = generate_date()
    dob_check = calculate_check_digit(dob)
    sex = random.choice(["M", "F", "<"])
    expiry = generate_date()
    exp_check = calculate_check_digit(expiry)
    personal = pad(random_mrz_char(random.randint(0, 14)), 14)
    personal_check = calculate_check_digit(personal)
    composite_data = doc_num + doc_check + nationality + dob + dob_check + sex + expiry + exp_check + personal + personal_check
    composite_check = calculate_check_digit(composite_data)
    line2 = f"{doc_num}{doc_check}{nationality}{dob}{dob_check}{sex}{expiry}{exp_check}{personal}{personal_check}{composite_check}"

    return line1 + "\n" + line2

def generate_td1():
    doc_type = random.choice(["I", "A", "C"]) + random.choice([random_alpha(1), "<"])
    country = random_alpha(3)
    doc_num = random_mrz_char(9)
    doc_check = calculate_check_digit(doc_num)
    optional1 = pad(random_mrz_char(random.randint(0, 15)), 15)
    line1 = f"{doc_type}{country}{doc_num}{doc_check}{optional1}"

    dob = generate_date()
    dob_check = calculate_check_digit(dob)
    sex = random.choice(["M", "F", "<"])
    expiry = generate_date()
    exp_check = calculate_check_digit(expiry)
    nationality = random_alpha(3)
    optional2 = pad(random_mrz_char(random.randint(0, 11)), 11)
    composite_data = line1[5:30] + dob + dob_check + expiry + exp_check + optional2
    composite_check = calculate_check_digit(composite_data)
    line2 = f"{dob}{dob_check}{sex}{expiry}{exp_check}{nationality}{optional2}{composite_check}"

    name = pad(generate_name(), 30)
    line3 = name

    return line1 + "\n" + line2 + "\n" + line3

def generate_td2():
    doc_type = random.choice(["I", "A", "C"]) + random.choice([random_alpha(1), "<"])
    country = random_alpha(3)
    name = pad(generate_name(), 31)
    line1 = f"{doc_type}{country}{name}"

    doc_num = random_mrz_char(9)
    doc_check = calculate_check_digit(doc_num)
    nationality = random_alpha(3)
    dob = generate_date()
    dob_check = calculate_check_digit(dob)
    sex = random.choice(["M", "F", "<"])
    expiry = generate_date()
    exp_check = calculate_check_digit(expiry)
    optional = pad(random_mrz_char(random.randint(0, 7)), 7)
    composite_data = doc_num + doc_check + dob + dob_check + expiry + exp_check + optional
    composite_check = calculate_check_digit(composite_data)
    line2 = f"{doc_num}{doc_check}{nationality}{dob}{dob_check}{sex}{expiry}{exp_check}{optional}{composite_check}"

    return line1 + "\n" + line2

if __name__ == "__main__":
    with open("ground-truth/training_text.txt", "w") as f:
        for _ in range(2000):
            fmt = random.choice([generate_td1, generate_td2, generate_td3])
            f.write(fmt() + "\n\n")
    print("Generated 2000 MRZ samples in ground-truth/training_text.txt")
```

Run it:

```bash
python3 generate_training_text.py
```

### 4.2 Render Text to Images

Use Tesseract's `text2image` tool to create image/ground-truth pairs:

```bash
FONTS_DIR="/Library/Fonts"  # macOS; use /usr/share/fonts on Linux

# Generate at multiple exposure levels to vary contrast
for exposure in -2 -1 0 1 2 3; do
  text2image \
    --text=ground-truth/training_text.txt \
    --outputbase=ground-truth/mrz_exp${exposure} \
    --font="OCR-B" \
    --fonts_dir="$FONTS_DIR" \
    --exposure=$exposure \
    --xsize=3600 \
    --ysize=200 \
    --char_spacing=0.5 \
    --leading=32 \
    --margin=20
done
```

This produces `.tif` image files and `.box` coordinate files for each exposure level.

### 4.3 Add Real-World Samples (Optional but Recommended)

If you have real photos of MRZ zones from passports or ID cards:

1. Crop just the MRZ region from each photo
2. Save as `.tif` or `.png`
3. Create matching `.gt.txt` files with the exact MRZ text
4. Place them in `ground-truth/`

Real-world samples dramatically improve accuracy because they contain authentic noise, lighting, and print quality variations.

---

## 5. Image Augmentation

This is the single most impactful improvement. Camera-captured MRZ images differ from synthetic images in many ways. Augmentation bridges this gap.

**File: `augment_images.py`**

```python
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def add_gaussian_noise(img, mean=0, std_range=(5, 25)):
    std = np.random.uniform(*std_range)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = int(amount * img.size * 0.5)
    num_pepper = int(amount * img.size * 0.5)
    # Salt
    coords = tuple(np.random.randint(0, max(1, d), num_salt) for d in img.shape)
    noisy[coords] = 255
    # Pepper
    coords = tuple(np.random.randint(0, max(1, d), num_pepper) for d in img.shape)
    noisy[coords] = 0
    return noisy

def apply_blur(img):
    blur_type = np.random.choice(["gaussian", "motion", "none"], p=[0.4, 0.3, 0.3])
    if blur_type == "gaussian":
        ksize = np.random.choice([3, 5])
        sigma = np.random.uniform(0.5, 2.0)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif blur_type == "motion":
        size = np.random.choice([3, 5, 7])
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = np.ones(size) / size
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        kernel = kernel / kernel.sum()
        return cv2.filter2D(img, -1, kernel)
    return img

def adjust_brightness_contrast(img):
    alpha = np.random.uniform(0.6, 1.4)  # contrast
    beta = np.random.uniform(-40, 40)     # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_rotation(img):
    angle = np.random.uniform(-5, 5)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # Use white background for MRZ
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def apply_perspective_warp(img):
    h, w = img.shape[:2]
    offset = int(min(w, h) * 0.03)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [np.random.randint(0, offset + 1), np.random.randint(0, offset + 1)],
        [w - np.random.randint(0, offset + 1), np.random.randint(0, offset + 1)],
        [w - np.random.randint(0, offset + 1), h - np.random.randint(0, offset + 1)],
        [np.random.randint(0, offset + 1), h - np.random.randint(0, offset + 1)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def add_uneven_lighting(img):
    h, w = img.shape[:2]
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    gradient = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - y) ** 2 + (j - x) ** 2)
            gradient[i, j] = 1.0 - min(dist / max(w, h), 0.4)
    intensity = np.random.uniform(20, 60)
    if len(img.shape) == 3:
        gradient = np.stack([gradient] * 3, axis=-1)
    result = img.astype(np.float32) + gradient * intensity
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_jpeg_compression(img):
    quality = np.random.randint(30, 80)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

def augment_image(img):
    """Apply a random combination of augmentations."""
    if np.random.random() > 0.3:
        img = adjust_brightness_contrast(img)
    if np.random.random() > 0.3:
        img = apply_blur(img)
    if np.random.random() > 0.4:
        img = add_gaussian_noise(img)
    if np.random.random() > 0.7:
        img = add_salt_pepper_noise(img)
    if np.random.random() > 0.4:
        img = apply_rotation(img)
    if np.random.random() > 0.6:
        img = apply_perspective_warp(img)
    if np.random.random() > 0.6:
        img = add_uneven_lighting(img)
    if np.random.random() > 0.5:
        img = apply_jpeg_compression(img)
    return img

def main():
    input_dir = "ground-truth"
    output_dir = "augmented"
    augmentations_per_image = 5

    os.makedirs(output_dir, exist_ok=True)

    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    print(f"Found {len(tif_files)} source images")

    for tif_path in tqdm(tif_files):
        img = cv2.imread(tif_path)
        if img is None:
            continue

        basename = os.path.splitext(os.path.basename(tif_path))[0]
        box_path = tif_path.replace(".tif", ".box")

        for i in range(augmentations_per_image):
            aug_img = augment_image(img.copy())
            out_name = f"{basename}_aug{i}"
            cv2.imwrite(os.path.join(output_dir, f"{out_name}.tif"), aug_img)

            # Copy the box file (coordinates stay the same for small augmentations)
            if os.path.exists(box_path):
                with open(box_path, "r") as f:
                    box_data = f.read()
                with open(os.path.join(output_dir, f"{out_name}.box"), "w") as f:
                    f.write(box_data)

    total = len(tif_files) * augmentations_per_image
    print(f"Generated {total} augmented images in {output_dir}/")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python3 augment_images.py
```

### Augmentation Summary

| Augmentation          | What It Simulates                          |
|-----------------------|--------------------------------------------|
| Gaussian noise        | Camera sensor noise in low light           |
| Salt & pepper noise   | Dead pixels, dust on lens                  |
| Gaussian blur         | Camera out of focus                        |
| Motion blur           | Hand shake during capture                  |
| Brightness/contrast   | Different lighting environments            |
| Rotation              | Document not perfectly aligned             |
| Perspective warp      | Document photographed at an angle          |
| Uneven lighting       | Shadow or light falling across document    |
| JPEG compression      | Image quality loss from compression        |

---

## 6. Training Pipeline

### 6.1 Extract the Base Model

Start by extracting the LSTM network from your existing model to use as a starting point (fine-tuning):

```bash
# Decompress the existing model
gunzip -k model/mrz.traineddata.gz

# Extract the LSTM component
combine_tessdata -e model/mrz.traineddata output/mrz.lstm
```

### 6.2 Generate LSTM Training Data

Convert each image/box pair into Tesseract's `.lstmf` training format:

```bash
# Process original images
for tif in ground-truth/*.tif; do
  base=$(basename "$tif" .tif)
  tesseract "$tif" "ground-truth/$base" lstm.train \
    --psm 6 \
    -l mrz
done

# Process augmented images
for tif in augmented/*.tif; do
  base=$(basename "$tif" .tif)
  tesseract "$tif" "augmented/$base" lstm.train \
    --psm 6 \
    -l mrz
done
```

The `--psm 6` flag tells Tesseract to assume a uniform block of text, which matches MRZ layout.

### 6.3 Create the Training File List

```bash
find ground-truth -name "*.lstmf" > output/training_files.txt
find augmented -name "*.lstmf" >> output/training_files.txt

# Verify
echo "Total training files: $(wc -l < output/training_files.txt)"
```

You should aim for at least 1000 `.lstmf` files. More is better.

### 6.4 Create an Eval File List (Optional but Recommended)

Set aside 10-20% of files for evaluation:

```bash
# Shuffle and split
shuf output/training_files.txt > output/all_files.txt
total=$(wc -l < output/all_files.txt)
eval_count=$((total / 10))
train_count=$((total - eval_count))

head -n $train_count output/all_files.txt > output/train_files.txt
tail -n $eval_count output/all_files.txt > output/eval_files.txt

echo "Training: $train_count files"
echo "Evaluation: $eval_count files"
```

### 6.5 Run Fine-Tuning

```bash
lstmtraining \
  --model_output=output/mrz_improved \
  --continue_from=output/mrz.lstm \
  --traineddata=model/mrz.traineddata \
  --train_listfile=output/train_files.txt \
  --eval_listfile=output/eval_files.txt \
  --max_iterations=10000 \
  --target_error_rate=0.001 \
  --debug_interval=500 \
  --learning_rate=0.0001 \
  --net_mode=192
```

**Parameter reference:**

| Parameter              | Value    | Explanation                                            |
|------------------------|----------|--------------------------------------------------------|
| `--max_iterations`     | 10000    | Training steps; increase if error is still decreasing  |
| `--target_error_rate`  | 0.001    | Stop early if error drops below 0.1%                  |
| `--debug_interval`     | 500      | Print progress every 500 iterations                    |
| `--learning_rate`      | 0.0001   | Low rate for fine-tuning (prevents forgetting)         |
| `--net_mode`           | 192      | Adam optimizer (recommended for fine-tuning)           |

Training will output checkpoint files: `mrz_improved_checkpoint`, `mrz_improved_N.N_M.checkpoint` (where N.N is the error rate).

### 6.6 Monitor Training Progress

Watch for these in the training output:

```
At iteration 500, stage 0, Eval Char error rate=2.35, Word error rate=8.12
At iteration 1000, stage 0, Eval Char error rate=0.82, Word error rate=3.45
At iteration 2000, stage 0, Eval Char error rate=0.15, Word error rate=0.89
```

- **Char error rate** should steadily decrease. Below 1% is good. Below 0.1% is excellent.
- If the error rate plateaus for more than 2000 iterations, training is done.
- If the error rate increases, the learning rate may be too high.

### 6.7 Export the Final Model

```bash
# Use the best checkpoint (lowest error rate)
lstmtraining --stop_training \
  --continue_from=output/mrz_improved_checkpoint \
  --traineddata=model/mrz.traineddata \
  --model_output=output/mrz.traineddata
```

---

## 7. Evaluating the Model

### 7.1 Quick Test with Known MRZ Strings

Create a test image and check recognition:

```bash
# Create a test image from known MRZ text
echo "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<" > eval/test.txt
echo "L898902C36UTO7408122F1204159ZE184226B<<<<<10" >> eval/test.txt

text2image \
  --text=eval/test.txt \
  --outputbase=eval/test \
  --font="OCR-B" \
  --fonts_dir=/Library/Fonts

# Run recognition with old model
tesseract eval/test.tif eval/result_old -l mrz --psm 6
echo "=== Old Model ==="
cat eval/result_old.txt

# Run recognition with new model
tesseract eval/test.tif eval/result_new --tessdata-dir output -l mrz --psm 6
echo "=== New Model ==="
cat eval/result_new.txt
```

### 7.2 Accuracy Benchmark Script

**File: `evaluate_model.py`**

```python
import subprocess
import os
import glob

def run_tesseract(image_path, tessdata_dir, lang="mrz"):
    out_base = image_path.replace(".tif", "_result")
    subprocess.run([
        "tesseract", image_path, out_base,
        "--tessdata-dir", tessdata_dir,
        "-l", lang,
        "--psm", "6"
    ], capture_output=True)
    result_file = out_base + ".txt"
    if os.path.exists(result_file):
        with open(result_file) as f:
            return f.read().strip()
    return ""

def char_accuracy(predicted, expected):
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    total = max(len(expected), 1)
    return correct / total * 100

def evaluate(test_dir, tessdata_dir):
    gt_files = sorted(glob.glob(os.path.join(test_dir, "*.gt.txt")))
    total_chars = 0
    correct_chars = 0

    for gt_file in gt_files:
        with open(gt_file) as f:
            expected = f.read().strip()

        img_file = gt_file.replace(".gt.txt", ".tif")
        if not os.path.exists(img_file):
            continue

        predicted = run_tesseract(img_file, tessdata_dir)
        predicted_clean = predicted.replace("\n", "").replace(" ", "")
        expected_clean = expected.replace("\n", "").replace(" ", "")

        for p, e in zip(predicted_clean, expected_clean):
            total_chars += 1
            if p == e:
                correct_chars += 1
        total_chars += abs(len(predicted_clean) - len(expected_clean))

    accuracy = correct_chars / max(total_chars, 1) * 100
    print(f"Character accuracy: {accuracy:.2f}% ({correct_chars}/{total_chars})")
    return accuracy

if __name__ == "__main__":
    print("=== Old Model ===")
    evaluate("eval", "model")

    print("=== New Model ===")
    evaluate("eval", "output")
```

### 7.3 Test with Augmented (Noisy) Images

The real test is recognition accuracy on noisy, camera-like images:

```bash
# Generate noisy test images
python3 -c "
import cv2, numpy as np, glob
for f in glob.glob('eval/*.tif'):
    img = cv2.imread(f)
    img = cv2.GaussianBlur(img, (3,3), 1.0)
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(f.replace('.tif', '_noisy.tif'), img)
"

# Test both models on noisy images
for img in eval/*_noisy.tif; do
  echo "--- $img ---"
  tesseract "$img" - -l mrz --psm 6 --tessdata-dir model 2>/dev/null
  echo "---"
  tesseract "$img" - -l mrz --psm 6 --tessdata-dir output 2>/dev/null
  echo ""
done
```

---

## 8. Packaging for Web

### 8.1 Compress the Model

```bash
gzip -9 -k output/mrz.traineddata
ls -lh output/mrz.traineddata.gz
```

### 8.2 Replace the Existing Model

```bash
# Back up the old model
cp model/mrz.traineddata.gz model/mrz.traineddata.gz.bak

# Install the new model
cp output/mrz.traineddata.gz model/mrz.traineddata.gz
```

### 8.3 Verify in the Browser

No code changes are needed in `index.js` since it already loads from `model/mrz.traineddata.gz` via Tesseract.js. Open your app and test with physical documents.

---

## 9. Troubleshooting

### Error: "No LSTM training data found"

The `.lstmf` files were not generated. Make sure `tesseract ... lstm.train` ran without errors and that the language data (`-l mrz`) is accessible.

### Error: "Char error rate stuck at high value"

- Increase `--max_iterations` to 20000 or more
- Check that your ground-truth text exactly matches the training images
- Ensure the augmented images are not too distorted to be readable

### Error: "Error rate increasing during training"

The learning rate is too high for fine-tuning. Reduce `--learning_rate` from `0.0001` to `0.00005` or `0.00001`.

### Error: "combine_tessdata: not found"

Training tools are not installed. Reinstall Tesseract:

```bash
# macOS
brew reinstall tesseract

# Ubuntu
sudo apt install tesseract-ocr-tools
```

### Model file size is much larger than the original

This is normal. You can reduce it by:

```bash
# Quantize the model (reduces size ~50%, minor accuracy tradeoff)
lstmtraining --stop_training \
  --continue_from=output/mrz_improved_checkpoint \
  --traineddata=model/mrz.traineddata \
  --model_output=output/mrz.traineddata \
  --convert_to_int
```

The `--convert_to_int` flag converts floating-point weights to integers, roughly halving the file size.

### Training from Scratch (Instead of Fine-Tuning)

If the existing model is too far from your target domain, you can train from scratch. This requires more data (5000+ images) and more iterations (50000+):

```bash
lstmtraining \
  --model_output=output/mrz_scratch \
  --traineddata=model/mrz.traineddata \
  --train_listfile=output/train_files.txt \
  --eval_listfile=output/eval_files.txt \
  --max_iterations=50000 \
  --target_error_rate=0.001 \
  --learning_rate=0.001 \
  --net_spec='[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c37]'
```

The `--net_spec` defines the LSTM network architecture:
- `Ct3,3,16`: Convolutional layer, 3x3 kernel, 16 filters
- `Mp3,3`: Max pooling, 3x3
- `Lfys48`: Full-height LSTM, 48 units
- `Lfx96` / `Lrx96`: Forward and reverse LSTM, 96 units
- `Lfx256`: Forward LSTM, 256 units
- `O1c37`: Output layer, 37 classes (26 letters + 10 digits + filler)

---

## Quick Reference: Complete Pipeline

```bash
# 1. Generate training text
python3 generate_training_text.py

# 2. Render to images
for exp in -2 -1 0 1 2 3; do
  text2image --text=ground-truth/training_text.txt \
    --outputbase=ground-truth/mrz_exp${exp} \
    --font="OCR-B" --fonts_dir=/Library/Fonts \
    --exposure=$exp --xsize=3600 --ysize=200
done

# 3. Augment images
python3 augment_images.py

# 4. Generate LSTM training data
for tif in ground-truth/*.tif augmented/*.tif; do
  base="${tif%.tif}"
  tesseract "$tif" "$base" lstm.train --psm 6 -l mrz
done

# 5. Build file lists
find ground-truth augmented -name "*.lstmf" | shuf > output/all.txt
head -n $(( $(wc -l < output/all.txt) * 9 / 10 )) output/all.txt > output/train.txt
tail -n +$(( $(wc -l < output/all.txt) * 9 / 10 + 1 )) output/all.txt > output/eval.txt

# 6. Extract base model
gunzip -k model/mrz.traineddata.gz
combine_tessdata -e model/mrz.traineddata output/mrz.lstm

# 7. Fine-tune
lstmtraining \
  --model_output=output/mrz_improved \
  --continue_from=output/mrz.lstm \
  --traineddata=model/mrz.traineddata \
  --train_listfile=output/train.txt \
  --eval_listfile=output/eval.txt \
  --max_iterations=10000 \
  --target_error_rate=0.001 \
  --learning_rate=0.0001

# 8. Export
lstmtraining --stop_training \
  --continue_from=output/mrz_improved_checkpoint \
  --traineddata=model/mrz.traineddata \
  --model_output=output/mrz.traineddata

# 9. Package
gzip -9 -k output/mrz.traineddata
cp output/mrz.traineddata.gz model/mrz.traineddata.gz
```
