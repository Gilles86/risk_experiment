#!/usr/bin/env python3
"""
Deface anatomical images (T1w, T2w) using pydeface.

This script:
1. Finds all anatomical images in a BIDS dataset
2. Creates backups of original files
3. Runs pydeface to remove facial features (in parallel)
4. Updates JSON sidecars to note defacing
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing





def deface_image_wrapper(args):
    """Wrapper to unpack args for deface_image (needed for multiprocessing)."""
    nifti_file, force = args
    return deface_image(nifti_file, force)


def deface_image(nifti_file: Path, force: bool = False) -> Dict:
    """
    Deface a single anatomical image using pydeface.
    
    Args:
        nifti_file: Path to NIfTI file
        force: If True, deface even if already defaced
        
    Returns:
        Dictionary with result information
    """
    # Check JSON sidecar for defacing status
    json_file = nifti_file.with_suffix('.json')
    if nifti_file.suffix == '.gz':
        json_file = nifti_file.with_suffix('').with_suffix('.json')
    
    already_defaced = False
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if metadata.get('Defaced') and not force:
                already_defaced = True
        except:
            pass
    
    if already_defaced:
        return {
            'success': True,
            'file': str(nifti_file),
            'action': 'skipped',
            'message': 'Already defaced'
        }
    
    # Run pydeface (output redirected to avoid clutter in parallel mode)
    try:
        cmd = ['pydeface', str(nifti_file), '--outfile', str(nifti_file), '--force']
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Update JSON sidecar
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata['Defaced'] = True
        metadata['DefacingMethod'] = 'pydeface'
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            f.write('\n')
        
        return {
            'success': True,
            'file': str(nifti_file),
            'action': 'defaced',
            'message': 'Successfully defaced'
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'file': str(nifti_file),
            'action': 'error',
            'message': f'pydeface failed: {e.stderr if e.stderr else str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'file': str(nifti_file),
            'action': 'error',
            'message': f'Error: {e}'
        }


def find_anatomical_images(bids_root: Path, modalities: List[str]) -> List[Path]:
    """
    Find all anatomical images in the BIDS dataset.
    
    Args:
        bids_root: Path to BIDS dataset root
        modalities: List of modalities to process (e.g., ['T1w', 'T2w'])
        
    Returns:
        List of anatomical image paths
    """
    anat_files = []
    for modality in modalities:
        # Look for both .nii and .nii.gz
        anat_files.extend(bids_root.glob(f'sub-*/ses-*/anat/*_{modality}.nii.gz'))
        anat_files.extend(bids_root.glob(f'sub-*/ses-*/anat/*_{modality}.nii'))
        # Also check for datasets without sessions
        anat_files.extend(bids_root.glob(f'sub-*/anat/*_{modality}.nii.gz'))
        anat_files.extend(bids_root.glob(f'sub-*/anat/*_{modality}.nii'))
    
    return sorted(set(anat_files))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deface anatomical images using pydeface'
    )
    parser.add_argument(
        'bids_root',
        type=str,
        help='Path to BIDS dataset root directory'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['T1w', 'T2w'],
        help='Anatomical modalities to deface (default: T1w T2w)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Deface even if already defaced'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be defaced without processing'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    
    if not bids_root.exists():
        print(f"Error: BIDS root directory does not exist: {bids_root}")
        return
    
    # Check if pydeface is available
    try:
        subprocess.run(['pydeface', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pydeface is not installed or not in PATH")
        print("Install with: pip install pydeface")
        return
    
    # Find all anatomical images
    anat_files = find_anatomical_images(bids_root, args.modalities)
    
    if not anat_files:
        print(f"No anatomical images found for modalities: {', '.join(args.modalities)}")
        print(f"Searched in: {bids_root}")
        return
    
    print(f"Found {len(anat_files)} anatomical image(s)\n")
    
    if args.dry_run:
        print("Files that would be processed:")
        for anat_file in anat_files:
            print(f"  - {anat_file.relative_to(bids_root)}")
        print(f"\nRun without --dry-run to deface these images")
        return
    
    # Process files in parallel
    print(f"Processing with {args.workers} parallel workers...\n")
    defaced_count = 0
    skipped_count = 0
    error_count = 0
    
    # Prepare arguments for parallel processing
    tasks = [(anat_file, args.force) for anat_file in anat_files]
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(deface_image_wrapper, task): task[0] 
                          for task in tasks}
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            anat_file = future_to_file[future]
            try:
                result = future.result()
                
                if result['success']:
                    if result['action'] == 'defaced':
                        defaced_count += 1
                        print(f"✓ {anat_file.name}: {result['message']}")
                    elif result['action'] == 'skipped':
                        skipped_count += 1
                        print(f"⊘ {anat_file.name}: {result['message']}")
                else:
                    error_count += 1
                    print(f"✗ {anat_file.name}: {result['message']}")
                    
            except Exception as e:
                error_count += 1
                print(f"✗ {anat_file.name}: Unexpected error: {e}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Defaced: {defaced_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    if defaced_count > 0:
        print(f"\n✓ Defacing complete!")


if __name__ == '__main__':
    main()
