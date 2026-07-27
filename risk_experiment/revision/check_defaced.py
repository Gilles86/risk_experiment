#!/usr/bin/env python3
"""
Check that all anatomical images in a BIDS dataset have been defaced.
Reports any missing or non-defaced files.
"""

import json
import argparse
from pathlib import Path


def check_defaced_status(dataset_path):
    """Check defacing status of all anatomical images."""
    dataset_path = Path(dataset_path)
    
    # Find all anatomical NIfTI files
    anat_files = []
    anat_files.extend(dataset_path.glob('**/anat/*_T1w.nii.gz'))
    anat_files.extend(dataset_path.glob('**/anat/*_T1w.nii'))
    anat_files.extend(dataset_path.glob('**/anat/*_T2w.nii.gz'))
    anat_files.extend(dataset_path.glob('**/anat/*_T2w.nii'))
    
    total_files = len(anat_files)
    defaced_files = []
    not_defaced_files = []
    missing_json_files = []
    missing_defaced_field = []
    
    print(f"\n{'='*70}")
    print(f"DEFACING STATUS CHECK")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"Total anatomical files found: {total_files}")
    print()
    
    for nifti_file in sorted(anat_files):
        # Get corresponding JSON file
        if nifti_file.suffix == '.gz':
            json_file = nifti_file.with_suffix('').with_suffix('.json')
        else:
            json_file = nifti_file.with_suffix('.json')
        
        if not json_file.exists():
            missing_json_files.append(nifti_file)
            continue
        
        # Check JSON for Defaced field
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'Defaced' not in data:
            missing_defaced_field.append(nifti_file)
        elif data['Defaced'] is True:
            defaced_files.append(nifti_file)
        else:
            not_defaced_files.append(nifti_file)
    
    # Print summary
    print(f"✓ Defaced: {len(defaced_files)}")
    print(f"✗ Not defaced: {len(not_defaced_files)}")
    print(f"⚠ Missing 'Defaced' field: {len(missing_defaced_field)}")
    print(f"⚠ Missing JSON sidecar: {len(missing_json_files)}")
    print()
    
    # Print details if there are problems
    if not_defaced_files:
        print(f"{'='*70}")
        print(f"FILES MARKED AS NOT DEFACED:")
        print(f"{'='*70}")
        for f in not_defaced_files:
            print(f"  {f.relative_to(dataset_path)}")
        print()
    
    if missing_defaced_field:
        print(f"{'='*70}")
        print(f"FILES MISSING 'Defaced' FIELD:")
        print(f"{'='*70}")
        for f in missing_defaced_field:
            print(f"  {f.relative_to(dataset_path)}")
        print()
    
    if missing_json_files:
        print(f"{'='*70}")
        print(f"FILES MISSING JSON SIDECAR:")
        print(f"{'='*70}")
        for f in missing_json_files:
            print(f"  {f.relative_to(dataset_path)}")
        print()
    
    # Final verdict
    print(f"{'='*70}")
    if len(defaced_files) == total_files:
        print(f"✓ SUCCESS: All {total_files} anatomical files are defaced!")
    else:
        print(f"⚠ WARNING: {total_files - len(defaced_files)} file(s) need attention")
    print(f"{'='*70}")
    print()
    
    return len(defaced_files) == total_files


def main():
    parser = argparse.ArgumentParser(
        description='Check defacing status of anatomical images in a BIDS dataset'
    )
    parser.add_argument(
        'bids_root',
        type=str,
        help='Path to the BIDS dataset root directory'
    )
    
    args = parser.parse_args()
    
    all_defaced = check_defaced_status(args.bids_root)
    
    # Exit with appropriate code
    exit(0 if all_defaced else 1)


if __name__ == '__main__':
    main()
