"""
receipt_ocr_final.py - READY TO USE Receipt OCR

NO encoding issues, NO complex dependencies, JUST WORKS.

Install:
    pip install easyocr opencv-python pillow numpy

Usage:
    python receipt_ocr_final.py --image your_receipt.jpg
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import json
import argparse
from pathlib import Path
from typing import List, Dict


class ReceiptOCR:
    """Simple but powerful OCR for receipts"""

    def __init__(self, languages=['en', 'uk', 'ru']):
        """Initialize OCR with languages"""
        print("Loading EasyOCR...")
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=True, verbose=False)
            print(f"Ready! Languages: {', '.join(languages)}")
        except ImportError:
            print("ERROR: EasyOCR not installed!")
            print("Run: pip install easyocr")
            exit(1)

    def enhance_image(self, image):
        """Make bad images readable"""
        print("  Enhancing image quality...")

        # Convert to PIL
        pil_img = Image.fromarray(image)

        # 1. Contrast boost (makes text darker)
        contrast = ImageEnhance.Contrast(pil_img)
        pil_img = contrast.enhance(2.5)

        # 2. Brightness boost (makes background lighter)
        brightness = ImageEnhance.Brightness(pil_img)
        pil_img = brightness.enhance(1.3)

        # 3. Sharpness boost (makes edges clearer)
        sharpness = ImageEnhance.Sharpness(pil_img)
        pil_img = sharpness.enhance(2.0)

        # Convert back to numpy
        img = np.array(pil_img)

        # 4. Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # 5. Apply adaptive threshold (black text on white)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 10
        )

        # 6. Remove noise
        clean = cv2.fastNlMeansDenoising(binary, h=10)

        return clean

    def process_receipt(self, image_path, save_enhanced=True):
        """
        Process receipt image

        Args:
            image_path: Path to receipt image
            save_enhanced: Save enhanced image for debugging

        Returns:
            List of detected text with confidence and position
        """
        print(f"\nProcessing: {image_path}")

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"ERROR: Cannot load {image_path}")
            return []

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

        # Enhance image
        enhanced = self.enhance_image(img)

        # Save enhanced for debugging
        if save_enhanced:
            enhanced_path = Path(image_path).with_suffix('.enhanced.jpg')
            cv2.imwrite(str(enhanced_path), enhanced)
            print(f"  Saved enhanced: {enhanced_path.name}")

        # Run OCR
        print("  Running OCR...")
        results = self.reader.readtext(enhanced)

        # Format results
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
        return ocr_data

    def structure_receipt(self, ocr_data):
        """
        Try to identify receipt structure

        Returns dict with: items, total, cash, change
        """
        receipt = {
            'items': [],
            'total': None,
            'subtotal': None,
            'discount': None,
            'tax': None,
            'cash': None,
            'change': None
        }

        # Sort by Y position (top to bottom)
        sorted_data = sorted(ocr_data, key=lambda x: x['position']['y'])

        for i, item in enumerate(sorted_data):
            text = item['text']
            text_upper = text.upper()
            y = item['position']['y']

            # Helper: find numbers on same line
            def find_number_on_line(y_pos, tolerance=20):
                for other in sorted_data:
                    if abs(other['position']['y'] - y_pos) < tolerance:
                        other_text = other['text']
                        # Check if it's a number
                        if any(c.isdigit() for c in other_text):
                            # Clean number
                            clean = other_text.replace(',', '').replace(' ', '')
                            if clean.replace('.', '').replace('-', '').isdigit():
                                return other_text
                return None

            # Check for keywords
            if 'TOTAL' in text_upper and 'SUB' not in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['total'] = num

            elif 'SUBTOTAL' in text_upper or 'SUB TOTAL' in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['subtotal'] = num

            elif 'DISCOUNT' in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['discount'] = num

            elif 'TAX' in text_upper or 'PB1' in text_upper or 'VAT' in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['tax'] = num

            elif 'CASH' in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['cash'] = num

            elif 'CHANGE' in text_upper:
                num = find_number_on_line(y)
                if num:
                    receipt['change'] = num

            else:
                # Might be an item
                # Simple heuristic: if text is on left side and has letters
                if item['position']['x'] < 300 and any(c.isalpha() for c in text):
                    # Look for price on same line
                    price = find_number_on_line(y)
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
    parser = argparse.ArgumentParser(description='Receipt OCR - Ready to use')
    parser.add_argument('--image', type=str, required=True, help='Receipt image path')
    parser.add_argument('--output', type=str, help='Output JSON path (optional)')
    parser.add_argument('--lang', nargs='+', default=['en', 'uk', 'ru'],
                        help='Languages (default: en uk ru)')
    parser.add_argument('--no-enhance', action='store_true',
                        help='Skip image enhancement')

    args = parser.parse_args()

    # Create OCR
    ocr = ReceiptOCR(languages=args.lang)

    # Process
    results = ocr.process_receipt(args.image, save_enhanced=not args.no_enhance)

    if not results:
        print("\nNo text detected!")
        return

    # Try to structure
    print("\nStructuring receipt...")
    receipt = ocr.structure_receipt(results)

    # Prepare output
    output_data = {
        'receipt': receipt,
        'raw_ocr': results
    }

    # Save to JSON
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.image).with_suffix('.json')

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
        if len(receipt['items']) > 10:
            print(f"  ... and {len(receipt['items']) - 10} more")

    print(f"\nTotals:")
    if receipt['subtotal']:
        print(f"  Subtotal: {receipt['subtotal']}")
    if receipt['discount']:
        print(f"  Discount: {receipt['discount']}")
    if receipt['tax']:
        print(f"  Tax:      {receipt['tax']}")
    if receipt['total']:
        print(f"  TOTAL:    {receipt['total']}")
    if receipt['cash']:
        print(f"  Cash:     {receipt['cash']}")
    if receipt['change']:
        print(f"  Change:   {receipt['change']}")

    print(f"\nRaw OCR ({len(results)} detections):")
    for i, item in enumerate(results[:15], 1):
        conf = item['confidence']
        text = item['text']
        print(f"  {i:2d}. [{conf:.2f}] {text}")
    if len(results) > 15:
        print(f"  ... and {len(results) - 15} more")

    print("\n" + "=" * 60)
    print(f"Full results saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()