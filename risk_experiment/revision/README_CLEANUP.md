# BIDS Fixing Scripts - Cleanup Notes

## Main Script to Use
**`fix_all_bids.py`** - Complete BIDS fixing script that runs all fixes in order:
1. **Fixes IntendedFor paths** - Maps fieldmap EPI files to corresponding BOLD files
2. **Fixes events.tsv files** - Fills empty duration values and reorders columns
3. **Adds required JSON metadata** - Adds Manufacturer, MagneticFieldStrength, EchoTime, and FlipAngle
4. **Fixes NIfTI temporal units** - Corrects 7T files from 'msec' to 'sec' (CRITICAL!)
5. **Syncs JSON RepetitionTime to NIfTI headers** - Ensures exact match for validator
6. Removes macOS resource fork files

Run with: `python3 fix_all_bids.py`

## Essential Script
**`deface_anatomicals.py`** - Use AFTER dataset is validated:
```bash
python3 deface_anatomicals.py /data/ds-risk_bids2
```

## Issues Fixed

### 1. IntendedFor Paths (ERROR: INTENDED_FOR)
**Problem:** EPI fieldmap JSON files had incorrect paths like `...task-task_run-1_epi_bold.nii.gz`  
**Solution:** Fixed to correct format: `...task-task_run-1_bold.nii.gz`

### 2. Empty Duration Values in Events Files
**Problem:** events.tsv files had duration column but values were empty/NaN for stimulus events  
**Root cause:** Duration mapping used `'stimulus_1'` but actual trial_type values were `'stimulus 1'` (with spaces)  
**Solution:** Updated mapping to use correct keys and fill NaN values with proper durations (0.6s for stimulus, 1.0s for choice/certainty)

### 3. RepetitionTime Mismatch (ERROR: REPETITION_TIME_MISMATCH)
**Problem:** BIDS validator reported TR mismatch despite values appearing identical in Python  
**Root cause:** 7T NIfTI files had temporal units set to `'msec'` instead of `'sec'`, causing validator to misinterpret TR values  
**Solution:** Fixed NIfTI headers using `img.header.set_xyzt_units(spatial_units, 'sec')`

### 4. JSON Metadata
**Problem:** Missing required/recommended Manufacturer, MagneticFieldStrength, EchoTime, and FlipAngle fields  
**Solution:** Added to all JSON sidecars based on session type and acquisition parameters from the paper:

**3T Acquisitions:**
- Functional (BOLD/EPI): FlipAngle=90°, EchoTime=0.030s (30ms), MagneticFieldStrength=3T
- Anatomical (T1w): FlipAngle=8°, EchoTime=0.0037s (3.7ms), MagneticFieldStrength=3T

**7T Acquisitions:**
- Functional (BOLD/EPI): FlipAngle=74°, EchoTime=0.015s (15ms), MagneticFieldStrength=7T
- Anatomical (T1w): FlipAngle=7°, EchoTime=0.0045s (4.5ms), MagneticFieldStrength=7T

All files: Manufacturer="Philips"

## Key Learning
The REPETITION_TIME_MISMATCH error was NOT about TR values being wrong - they were correct! The issue was **temporal units** in the NIfTI header. The 7T files had `xyzt_units = ('mm', 'msec')` when they should have been `('mm', 'sec')`. This caused the BIDS validator to think the TR was in milliseconds when it was actually in seconds.

**Critical fix:** Always check `nib.load(file).header.get_xyzt_units()` to verify temporal units match the actual TR scale!
