#!/usr/bin/env python3
"""
Complete BIDS fixing script for Risk Experiment dataset.
Applies all necessary fixes in the correct order:
1. Fix IntendedFor paths and remove macOS files
2. Fix events.tsv structure and columns
3. Add required JSON metadata
4. Sync JSON RepetitionTime to match NIfTI headers (FINAL STEP)
"""

import argparse
import json
import subprocess
import nibabel as nib
from pathlib import Path
import pandas as pd

def fix_intendedfor_fields(dataset_path):
    """Fix IntendedFor paths in fmap JSON files."""
    print("\n" + "="*70)
    print("STEP 1: Fixing IntendedFor paths in fmap JSON files")
    print("="*70)
    
    fmap_jsons = list(dataset_path.glob('**/fmap/*.json'))
    # Filter out macOS resource fork files
    fmap_jsons = [f for f in fmap_jsons if not f.name.startswith('._')]
    
    for json_file in sorted(fmap_jsons):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'IntendedFor' in data:
            old_value = data['IntendedFor']
            
            # Extract session and run number from filename
            if 'ses-3t2' in str(json_file):
                session = 'ses-3t2'
            elif 'ses-7t2' in str(json_file):
                session = 'ses-7t2'
            else:
                continue
            
            # Extract run number and subject from epi filename
            # e.g., sub-13_ses-3t2_run-1_epi.json → sub-13, run-1 → 1
            base_name = json_file.stem
            if '_run-' in base_name:
                # Extract subject number from filename
                subject_num = base_name.split('_')[0]  # 'sub-13'
                # Split on _run- and get just the number part
                run_part = base_name.split('_run-')[1]  # '1_epi'
                run_num = run_part.split('_')[0]  # '1'
                # Map to corresponding bold file with correct subject
                # IntendedFor paths are relative to subject directory: ses-XXX/func/...
                new_value = f"{session}/func/{subject_num}_{session}_task-task_run-{run_num}_bold.nii.gz"
                
                data['IntendedFor'] = new_value
                
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=4)
                    f.write('\n')
                
                print(f"✓ {json_file.name}: → {new_value}")

def fix_events_files(dataset_path):
    """Fix events.tsv: add duration column and reorder columns."""
    print("\n" + "="*70)
    print("STEP 2: Fixing events.tsv files")
    print("="*70)
    
    events_files = list(dataset_path.glob('**/func/*_events.tsv'))
    # Filter out macOS resource fork files
    events_files = [f for f in events_files if not f.name.startswith('._')]
    
    # Duration mapping - note: trial_type values use spaces, not underscores
    duration_map = {
        'stimulus 1': 0.6,
        'stimulus 2': 0.6,
        'choice': 1.0,
        'certainty': 1.0,
    }
    
    for events_file in sorted(events_files):
        df = pd.read_csv(events_file, sep='\t')
        
        # Fill in duration column based on trial_type
        if 'duration' in df.columns:
            # Fill NaN values with duration based on trial_type
            df['duration'] = df['duration'].fillna(df['trial_type'].map(duration_map))
        else:
            # Add duration column if missing
            df['duration'] = df['trial_type'].map(duration_map)
        
        # Reorder columns: onset, duration, trial_type first
        cols = ['onset', 'duration', 'trial_type']
        cols.extend([c for c in df.columns if c not in cols])
        df = df[cols]
        
        df.to_csv(events_file, sep='\t', index=False)
        print(f"✓ {events_file.name}: fixed duration values, reordered columns")

def fix_json_metadata(dataset_path):
    """Add required JSON metadata fields."""
    print("\n" + "="*70)
    print("STEP 3: Adding required JSON metadata")
    print("="*70)
    
    json_files = list(dataset_path.glob('**/*.json'))
    # Filter out macOS resource fork files
    json_files = [f for f in json_files if not f.name.startswith('._')]
    
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        changed = False
        
        # Add Manufacturer if missing
        if 'Manufacturer' not in data:
            data['Manufacturer'] = 'Philips'
            changed = True
        
        # Determine if this is 3T or 7T
        is_7t = 'ses-7t' in str(json_file)
        is_3t = 'ses-3t' in str(json_file)
        
        # Determine file type
        is_func = '/func/' in str(json_file) or '/fmap/' in str(json_file)  # BOLD or EPI
        is_anat = '/anat/' in str(json_file)  # T1w
        
        # Add MagneticFieldStrength if missing
        if 'MagneticFieldStrength' not in data:
            if is_3t:
                data['MagneticFieldStrength'] = 3
                changed = True
            elif is_7t:
                data['MagneticFieldStrength'] = 7
                changed = True
        
        # Add EchoTime if missing (in seconds, not milliseconds!)
        if 'EchoTime' not in data:
            if is_3t and is_func:
                data['EchoTime'] = 0.030  # 30ms for 3T functional
                changed = True
            elif is_3t and is_anat:
                data['EchoTime'] = 0.0037  # 3.7ms for 3T anatomical
                changed = True
            elif is_7t and is_func:
                data['EchoTime'] = 0.015  # 15ms for 7T functional
                changed = True
            elif is_7t and is_anat:
                data['EchoTime'] = 0.0045  # 4.5ms for 7T anatomical
                changed = True
        
        # Add FlipAngle if missing (in degrees)
        if 'FlipAngle' not in data:
            if is_3t and is_func:
                data['FlipAngle'] = 90  # 90° for 3T functional
                changed = True
            elif is_3t and is_anat:
                data['FlipAngle'] = 8  # 8° for 3T anatomical
                changed = True
            elif is_7t and is_func:
                data['FlipAngle'] = 74  # 74° for 7T functional
                changed = True
            elif is_7t and is_anat:
                data['FlipAngle'] = 7  # 7° for 7T anatomical
                changed = True
        
        if changed:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
                f.write('\n')
            print(f"✓ {json_file.name}: added metadata fields")

def fix_xyzt_units(dataset_path):
    """Fix temporal units in NIfTI headers (7T files have 'msec' instead of 'sec')."""
    print("\n" + "="*70)
    print("STEP 4: Fixing temporal units in NIfTI headers")
    print("="*70)
    
    # Find all NIfTI files in 7T2 sessions
    nifti_files = list(dataset_path.glob('**/ses-7t2/**/*.nii.gz'))
    # Filter out macOS resource fork files
    nifti_files = [f for f in nifti_files if not f.name.startswith('._')]
    
    for nifti_file in sorted(nifti_files):
        try:
            img = nib.load(str(nifti_file))
            spatial_units, temporal_units = img.header.get_xyzt_units()
            
            # Only fix if temporal units are 'msec'
            if temporal_units == 'msec':
                # Set to 'sec'
                img.header.set_xyzt_units(spatial_units, 'sec')
                nib.save(img, str(nifti_file))
                print(f"✓ {nifti_file.name}: temporal units fixed (msec → sec)")
            
        except Exception as e:
            print(f"✗ {nifti_file.name}: ERROR - {e}")

def fix_incorrect_nifti_tr(dataset_path):
    """Fix incorrect TR values in NIfTI headers (e.g., TR=1.0 when should be ~2.3)."""
    print("\n" + "="*70)
    print("STEP 4b: Fixing incorrect TR values in NIfTI headers")
    print("="*70)
    
    # Expected TR values for each session type
    tr_expected = {
        'ses-3t1': 8.0,    # 3T anatomical
        'ses-3t2': 2.297924518585205,  # 3T functional
        'ses-7t1': 9.8,    # 7T anatomical
        'ses-7t2': 2.299999952316284,  # 7T functional
    }
    
    # Find all BOLD and EPI files
    nifti_files = list(dataset_path.glob('**/func/*_bold.nii.gz'))
    nifti_files.extend(dataset_path.glob('**/fmap/*_epi.nii.gz'))
    
    for nifti_file in sorted(nifti_files):
        try:
            img = nib.load(str(nifti_file))
            current_tr = float(img.header.structarr['pixdim'][4])
            
            # Determine expected TR from filename
            filename = str(nifti_file)
            expected_tr = None
            for session_key, tr_value in tr_expected.items():
                if session_key in filename:
                    expected_tr = tr_value
                    break
            
            if expected_tr is None:
                continue
            
            # Check if TR needs fixing (e.g., 1.0 when should be 2.3)
            if abs(current_tr - expected_tr) > 0.1:  # Tolerance for floating point
                print(f"✓ {nifti_file.name}: {current_tr} → {expected_tr}")
                # Fix the NIfTI header
                img.header.structarr['pixdim'][4] = expected_tr
                nib.save(img, str(nifti_file))
        
        except Exception as e:
            print(f"✗ {nifti_file.name}: ERROR - {e}")

def sync_json_tr_to_nifti(dataset_path):
    """FINAL STEP: Sync JSON RepetitionTime to match NIfTI headers exactly."""
    print("\n" + "="*70)
    print("STEP 5: Syncing JSON RepetitionTime to NIfTI headers")
    print("="*70)
    print("(This ensures the validator sees matching values)")
    print()
    
    json_files = list(dataset_path.glob('**/func/*_bold.json'))
    json_files.extend(dataset_path.glob('**/fmap/*_epi.json'))
    # Filter out macOS resource fork files
    json_files = [f for f in json_files if not f.name.startswith('._')]
    
    for json_file in sorted(json_files):
        nifti_file = json_file.parent / json_file.name.replace('.json', '.nii.gz')
        
        if not nifti_file.exists():
            print(f"✗ {json_file.name}: NIfTI file not found")
            continue
        
        try:
            # Read TR from NIfTI header
            img = nib.load(str(nifti_file))
            tr_nifti = float(img.header.structarr['pixdim'][4])
            
            # Update JSON with exact same value
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            old_tr = data.get('RepetitionTime', 'N/A')
            data['RepetitionTime'] = tr_nifti
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
                f.write('\n')
            
            print(f"✓ {json_file.name}: {old_tr} → {tr_nifti}")
        
        except Exception as e:
            print(f"✗ {json_file.name}: ERROR - {e}")

def clean_macos_files(dataset_path):
    """Remove macOS resource fork files."""
    print("\n" + "="*70)
    print("STEP 0: Removing macOS resource fork files")
    print("="*70)
    
    mac_files = list(dataset_path.glob('**/_*'))
    if mac_files:
        for f in mac_files:
            try:
                f.unlink()
                print(f"✓ Removed {f.name}")
            except:
                pass
    else:
        print("✓ No macOS files found")

def create_missing_anat_jsons(dataset_path):
    """Create minimal JSON sidecars for anatomical files that are missing them."""
    print("\n" + "="*70)
    print("STEP 0.5: Creating missing anatomical JSON sidecars")
    print("="*70)
    
    # Find all anatomical NIfTI files
    anat_files = []
    anat_files.extend(dataset_path.glob('**/anat/*_T1w.nii.gz'))
    anat_files.extend(dataset_path.glob('**/anat/*_T1w.nii'))
    anat_files.extend(dataset_path.glob('**/anat/*_T2w.nii.gz'))
    anat_files.extend(dataset_path.glob('**/anat/*_T2w.nii'))
    
    created_count = 0
    
    for nifti_file in sorted(anat_files):
        # Determine the JSON sidecar path
        if nifti_file.suffix == '.gz':
            json_file = nifti_file.with_suffix('').with_suffix('.json')
        else:
            json_file = nifti_file.with_suffix('.json')
        
        # Only create if it doesn't exist
        if not json_file.exists():
            # Create minimal JSON sidecar
            minimal_metadata = {}
            
            with open(json_file, 'w') as f:
                json.dump(minimal_metadata, f, indent=4)
                f.write('\n')
            
            print(f"✓ Created {json_file.name}")
            created_count += 1
    
    if created_count == 0:
        print("✓ All anatomical files already have JSON sidecars")
    else:
        print(f"✓ Created {created_count} JSON sidecar(s)")

def process_subject(subject_path):
    """Process a single subject directory."""
    print("\n" + "█"*70)
    print(f"PROCESSING: {subject_path.name}")
    print("█"*70)
    
    # Clean up macOS files FIRST to avoid processing issues
    clean_macos_files(subject_path)
    
    # Create missing anatomical JSON sidecars BEFORE adding metadata
    create_missing_anat_jsons(subject_path)
    
    fix_intendedfor_fields(subject_path)
    fix_events_files(subject_path)
    fix_json_metadata(subject_path)
    fix_xyzt_units(subject_path)
    fix_incorrect_nifti_tr(subject_path)  # Fix TR in NIfTI headers BEFORE syncing
    sync_json_tr_to_nifti(subject_path)
    
    print("\n" + "█"*70)
    print(f"✓ {subject_path.name} COMPLETE")
    print("█"*70)


def main():
    parser = argparse.ArgumentParser(
        description='Fix BIDS format issues in Risk Experiment dataset'
    )
    parser.add_argument(
        'bids_root',
        type=str,
        help='Path to BIDS dataset root directory'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Process only this subject (e.g., sub-02). If not specified, processes all subjects.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List subjects that would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    
    if not bids_root.exists():
        print(f"Error: BIDS root directory does not exist: {bids_root}")
        return
    
    # Find subjects to process
    if args.subject:
        # Process single subject
        subject_path = bids_root / args.subject
        if not subject_path.exists():
            print(f"Error: Subject directory does not exist: {subject_path}")
            return
        subjects = [subject_path]
    else:
        # Process all subjects
        subjects = sorted([d for d in bids_root.glob('sub-*') if d.is_dir()])
    
    if not subjects:
        print(f"No subjects found in {bids_root}")
        return
    
    print("\n" + "█"*70)
    print("BIDS FIXING SCRIPT - Risk Experiment Dataset")
    print("█"*70)
    print(f"Dataset: {bids_root}")
    print(f"Subjects to process: {len(subjects)}")
    
    if args.dry_run:
        print("\nSubjects that would be processed:")
        for subject in subjects:
            print(f"  - {subject.name}")
        print(f"\nRun without --dry-run to process these subjects")
        return
    
    # Process each subject
    for i, subject_path in enumerate(subjects, 1):
        print(f"\n{'='*70}")
        print(f"Subject {i}/{len(subjects)}: {subject_path.name}")
        print(f"{'='*70}")
        process_subject(subject_path)
    
    print("\n" + "█"*70)
    print("✓ ALL SUBJECTS COMPLETE")
    print("█"*70)
    print("\nNext steps:")
    print(f"1. Run BIDS validator on {bids_root}")
    print("2. If validated successfully, deface anatomicals with:")
    print(f"   python3 deface_anatomicals.py {bids_root}")
    print()

if __name__ == '__main__':
    main()
