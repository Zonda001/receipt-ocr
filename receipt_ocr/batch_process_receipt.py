"""
batch_process_receipts.py - Process multiple receipts at once

Usage:
    python batch_process_receipts.py --folder receipts_folder/
    python batch_process_receipts.py --images img1.jpg img2.jpg img3.jpg
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Import from our main OCR file
try:
    from receipt_ocr_final import ReceiptOCR
except ImportError:
    print("ERROR: receipt_ocr_final.py must be in same folder!")
    sys.exit(1)


def process_batch(image_paths, languages=['en', 'uk', 'ru'], output_dir='batch_results'):
    """Process multiple receipts"""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nProcessing {len(image_paths)} receipts...")
    print(f"Output directory: {output_path}")
    print("=" * 60)

    # Initialize OCR once
    ocr = ReceiptOCR(languages=languages)

    # Statistics
    stats = {
        'total': len(image_paths),
        'success': 0,
        'failed': 0,
        'total_text_regions': 0,
        'receipts_with_total': 0,
        'receipts_with_items': 0,
        'start_time': datetime.now().isoformat()
    }

    results_summary = []

    # Process each receipt
    for image_path in tqdm(image_paths, desc="Processing"):
        try:
            # OCR
            ocr_results = ocr.process_receipt(image_path, save_enhanced=False)

            if ocr_results:
                stats['success'] += 1
                stats['total_text_regions'] += len(ocr_results)

                # Structure
                receipt = ocr.structure_receipt(ocr_results)

                if receipt['total']:
                    stats['receipts_with_total'] += 1
                if receipt['items']:
                    stats['receipts_with_items'] += 1

                # Save individual result
                image_name = Path(image_path).stem
                result_file = output_path / f"{image_name}_result.json"

                output_data = {
                    'image': str(image_path),
                    'receipt': receipt,
                    'raw_ocr': ocr_results
                }

                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                # Add to summary
                results_summary.append({
                    'image': str(image_path),
                    'success': True,
                    'text_regions': len(ocr_results),
                    'has_total': receipt['total'] is not None,
                    'items_count': len(receipt['items'])
                })
            else:
                stats['failed'] += 1
                results_summary.append({
                    'image': str(image_path),
                    'success': False,
                    'error': 'No text detected'
                })

        except Exception as e:
            stats['failed'] += 1
            results_summary.append({
                'image': str(image_path),
                'success': False,
                'error': str(e)
            })

    # Save summary
    stats['end_time'] = datetime.now().isoformat()
    summary_data = {
        'statistics': stats,
        'results': results_summary
    }

    summary_file = output_path / 'batch_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images:        {stats['total']}")
    print(f"Successful:          {stats['success']} ({stats['success'] / stats['total'] * 100:.1f}%)")
    print(f"Failed:              {stats['failed']}")
    print(f"\nTotal text regions:  {stats['total_text_regions']}")
    print(f"Avg per receipt:     {stats['total_text_regions'] / max(stats['success'], 1):.1f}")
    print(f"\nReceipts with TOTAL: {stats['receipts_with_total']}")
    print(f"Receipts with items: {stats['receipts_with_items']}")
    print(f"\nResults saved to:    {output_path}")
    print(f"Summary:             {summary_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Batch process receipts')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', type=str, help='Folder with receipt images')
    group.add_argument('--images', nargs='+', help='List of image paths')

    parser.add_argument('--output', type=str, default='batch_results',
                        help='Output directory (default: batch_results)')
    parser.add_argument('--lang', nargs='+', default=['en', 'uk', 'ru'],
                        help='Languages (default: en uk ru)')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                        help='File pattern for --folder (default: *.jpg)')

    args = parser.parse_args()

    # Collect image paths
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"ERROR: Folder not found: {folder}")
            return

        # Find all images
        image_paths = []
        for pattern in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(folder.glob(pattern))
            image_paths.extend(folder.glob(pattern.upper()))

        image_paths = sorted(set(image_paths))

        if not image_paths:
            print(f"ERROR: No images found in {folder}")
            return

        print(f"Found {len(image_paths)} images in {folder}")
    else:
        image_paths = [Path(p) for p in args.images]

        # Check all exist
        missing = [p for p in image_paths if not p.exists()]
        if missing:
            print(f"ERROR: Images not found:")
            for p in missing:
                print(f"  - {p}")
            return

    # Process
    process_batch(image_paths, languages=args.lang, output_dir=args.output)


if __name__ == '__main__':
    main()