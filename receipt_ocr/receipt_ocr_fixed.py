"""
receipt_ocr_fixed.py - FIXED VERSION with smarter preprocessing

The problem: Too aggressive enhancement destroyed the text.
The solution: Smart quality detection + adaptive enhancement.

Usage:
    python receipt_ocr_fixed.py --image receipt.jpg
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class SmartReceiptOCR:
    """OCR with smart preprocessing based on image quality"""

    def __init__(self, languages=['en', 'uk', 'ru']):
        print("Loading EasyOCR...")
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=True, verbose=False)
            print(f"Ready! Languages: {', '.join(languages)}")
        except ImportError:
            print("ERROR: Install EasyOCR first: pip install easyocr")
            exit(1)

    def analyze_quality(self, image):
        """
        Analyze image quality to decide preprocessing strategy

        Returns: dict with quality metrics
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate metrics
        brightness = gray.mean()
        contrast = gray.std()

        # Detect if image is too bright (washed out)
        too_bright = brightness > 200

        # Detect if image has low contrast
        low_contrast = contrast < 50

        # Detect if already binary (black and white only)
        unique_vals = len(np.unique(gray))
        is_binary = unique_vals < 50

        quality = {
            'brightness': brightness,
            'contrast': contrast,
            'too_bright': too_bright,
            'low_contrast': low_contrast,
            'is_binary': is_binary,
            'needs_enhancement': (too_bright or low_contrast) and not is_binary
        }

        return quality

    def enhance_smart(self, image, quality):
        """
        Smart enhancement based on image quality

        Key insight: Your image was GOOD, we made it WORSE!
        So: only enhance if really needed.
        """
        print(f"  Image quality:")
        print(f"    Brightness: {quality['brightness']:.1f}")
        print(f"    Contrast: {quality['contrast']:.1f}")
        print(f"    Binary: {quality['is_binary']}")

        # If image is already good - DON'T enhance!
        if not quality['needs_enhancement']:
            print("  Image quality is GOOD - skipping aggressive enhancement")

            # Just slight improvement
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Slight denoise only
            clean = cv2.fastNlMeansDenoising(gray, h=5)
            return clean

        # Image needs enhancement
        print("  Image quality is BAD - applying enhancement")

        pil_img = Image.fromarray(image)

        # Moderate enhancement (not extreme!)
        if quality['low_contrast']:
            contrast = ImageEnhance.Contrast(pil_img)
            pil_img = contrast.enhance(1.8)  # Was 2.5, now 1.8

        if quality['too_bright']:
            brightness = ImageEnhance.Brightness(pil_img)
            pil_img = brightness.enhance(0.9)  # Darken slightly

        # Sharpness
        sharpness = ImageEnhance.Sharpness(pil_img)
        pil_img = sharpness.enhance(1.5)  # Was 2.0, now 1.5

        # Convert to numpy
        img = np.array(pil_img)

        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Adaptive threshold (only if needed)
        if quality['low_contrast']:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 5  # Was 15, 10 - less aggressive
            )
            gray = binary

        # Denoise
        clean = cv2.fastNlMeansDenoising(gray, h=7)

        return clean

    def process_receipt(self, image_path, save_debug=True):
        """Process receipt with smart preprocessing"""
        print(f"\nProcessing: {image_path}")

        # Load
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"ERROR: Cannot load {image_path}")
            return []

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

        # Analyze quality
        quality = self.analyze_quality(img)

        # Smart enhancement
        enhanced = self.enhance_smart(img, quality)

        # Save for debugging
        if save_debug:
            debug_path = Path(image_path).with_suffix('.debug.jpg')
            cv2.imwrite(str(debug_path), enhanced)
            print(f"  Saved debug: {debug_path.name}")

        # OCR - try BOTH original and enhanced
        print("  Running OCR...")

        # Try enhanced first
        results_enhanced = self.reader.readtext(enhanced)

        # If no good results, try original
        good_results = [r for r in results_enhanced if r[2] > 0.5]

        if len(good_results) < 5:
            print("  Enhanced image gave poor results, trying original...")
            results_original = self.reader.readtext(img)

            # Use whichever gave better results
            if len(results_original) > len(results_enhanced):
                results = results_original
                print(f"  Original better: {len(results_original)} vs {len(results_enhanced)}")
            else:
                results = results_enhanced
        else:
            results = results_enhanced

        # Format
        ocr_data = []
        for bbox, text, conf in results:
            x1 = min(p[0] for p in bbox)
            y1 = min(p[1] for p in bbox)
            x2 = max(p[0] for p in bbox)
            y2 = max(p[1] for p in bbox)

            ocr_data.append({
                'text': text,
                'confidence': round(float(conf), 3),
                'position': {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                }
            })

        print(f"  Found {len(ocr_data)} text regions")

        # Filter low confidence
        filtered = [r for r in ocr_data if r['confidence'] > 0.3]
        print(f"  High confidence: {len(filtered)} (>{0.3})")

        return filtered

    def structure_receipt(self, ocr_data):
        """Structure receipt data"""
        receipt = {
            'items': [],
            'total': None,
            'subtotal': None,
            'discount': None,
            'tax': None,
            'cash': None,
            'change': None
        }

        # Sort by Y
        sorted_data = sorted(ocr_data, key=lambda x: x['position']['y'])

        for item in sorted_data:
            text = item['text']
            text_upper = text.upper()
            y = item['position']['y']

            # Find number on same line
            def find_num(y_pos):
                for other in sorted_data:
                    if abs(other['position']['y'] - y_pos) < 30:
                        other_text = other['text']
                        # Must be numeric
                        if any(c.isdigit() for c in other_text):
                            # Clean
                            clean = other_text.replace(' ', '')
                            if sum(c.isdigit() for c in clean) >= len(clean) * 0.5:
                                return other_text
                return None

            # Check keywords
            if 'SUMA' in text_upper or 'СУМА' in text_upper:
                num = find_num(y)
                if num:
                    receipt['total'] = num

            elif 'TOTAL' in text_upper and 'SUB' not in text_upper:
                num = find_num(y)
                if num:
                    receipt['total'] = num

            elif 'SUBTOTAL' in text_upper or 'ПІДСУМА' in text_upper:
                num = find_num(y)
                if num:
                    receipt['subtotal'] = num

            elif 'PB' in text_upper or 'ПДВ' in text_upper or 'PDV' in text_upper:
                num = find_num(y)
                if num:
                    receipt['tax'] = num

            elif 'CASH' in text_upper or 'ГОТІВКА' in text_upper:
                num = find_num(y)
                if num:
                    receipt['cash'] = num

            elif 'CHANGE' in text_upper or 'РЕШТА' in text_upper:
                num = find_num(y)
                if num:
                    receipt['change'] = num

            elif 'ЗНИЖКА' in text_upper or 'DISCOUNT' in text_upper:
                num = find_num(y)
                if num:
                    receipt['discount'] = num

            else:
                # Might be item
                x = item['position']['x']
                if x < 400 and len(text) > 2:
                    if sum(c.isalpha() or c.isspace() for c in text) > len(text) * 0.5:
                        price = find_num(y)
                        receipt['items'].append({
                            'name': text,
                            'price': price
                        })

        return receipt

    def save_json(self, data, output_path):
        """Save to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Smart Receipt OCR')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, help='Output JSON')
    parser.add_argument('--lang', nargs='+', default=['en', 'uk', 'ru'])

    args = parser.parse_args()

    # OCR
    ocr = SmartReceiptOCR(languages=args.lang)
    results = ocr.process_receipt(args.image)

    if not results:
        print("\nNo text detected!")
        return

    # Structure
    print("\nStructuring receipt...")
    receipt = ocr.structure_receipt(results)

    # Output
    output_data = {
        'receipt': receipt,
        'raw_ocr': results
    }

    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.image).with_suffix('.fixed.json')

    ocr.save_json(output_data, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RECEIPT SUMMARY")
    print("=" * 60)

    if receipt['items']:
        print(f"\nItems ({len(receipt['items'])}):")
        for i, item in enumerate(receipt['items'][:10], 1):
            price = item.get('price', 'N/A')
            print(f"  {i}. {item['name']:30s} {price:>10s}")

    print(f"\nTotals:")
    for key in ['subtotal', 'discount', 'tax', 'total', 'cash', 'change']:
        if receipt[key]:
            print(f"  {key.capitalize():12s} {receipt[key]}")

    print(f"\nAll detected text ({len(results)}):")
    for i, item in enumerate(results[:20], 1):
        print(f"  {i:2d}. [{item['confidence']:.2f}] {item['text']}")

    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()