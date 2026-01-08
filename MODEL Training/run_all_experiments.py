"""
Master Script to Run All Feature Removal Experiments
This script runs all the individual feature removal scripts sequentially.
Each script will create its own folder with results.
"""

import subprocess
import sys
import os

# List of all scripts to run
scripts = [
    'Logistic_PM10_removed.py',
    'Logistic_NO2_removed.py',
    'Logistic_SO2_removed.py',
    'Logistic_CO_removed.py',
    'Logistic_O3_removed.py',
    'Logistic_Temperature_removed.py',
    'Logistic_Humidity_removed.py',
    'Logistic_WindSpeed_removed.py'
]

print("="*60)
print("  RUNNING ALL FEATURE REMOVAL EXPERIMENTS")
print("="*60)
print(f"\nTotal scripts to run: {len(scripts)}\n")

for i, script in enumerate(scripts, 1):
    print(f"\n{'='*60}")
    print(f"  [{i}/{len(scripts)}] Running: {script}")
    print(f"{'='*60}\n")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"\n✓ Successfully completed: {script}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script}")
        print(f"Error: {e}")
        continue
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script}")
        continue

print("\n" + "="*60)
print("  ALL EXPERIMENTS COMPLETED!")
print("="*60)
print("\nResults are saved in the following folders:")
print("  - PM10_removed/")
print("  - NO2_removed/")
print("  - SO2_removed/")
print("  - CO_removed/")
print("  - O3_removed/")
print("  - Temperature_removed/")
print("  - Humidity_removed/")
print("  - WindSpeed_removed/")
print("\nEach folder contains a text file with model evaluation results.")
