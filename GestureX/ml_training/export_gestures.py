"""
export_gestures.py - Export gesture examples to gestures.json

Reads collected CSV data and exports landmark examples to gestures.json
in the format: { "GESTURE_NAME": {"description": "...", "examples": [[63 floats], ...]} }

Usage:
    python export_gestures.py --input data/raw --output gestures.json --samples 5
"""

import argparse
import os
import json
from typing import Dict, List, Any

import pandas as pd
import numpy as np


def load_csv_data(input_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Load gesture data from CSV files.
    
    Args:
        input_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping gesture names to list of landmark arrays
    """
    data = {}
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return data
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return data
    
    print(f"Found {len(csv_files)} CSV files")
    
    for filename in csv_files:
        filepath = os.path.join(input_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            
            # Get landmark columns (lm_0 through lm_62)
            lm_cols = [col for col in df.columns if col.startswith('lm_')]
            
            if not lm_cols or 'label' not in df.columns:
                print(f"Skipping {filename}: missing landmark or label columns")
                continue
            
            # Group by label
            for label in df['label'].unique():
                label_data = df[df['label'] == label][lm_cols].values
                
                if label not in data:
                    data[label] = []
                
                data[label].extend([row.tolist() for row in label_data])
            
            print(f"  Loaded {len(df)} samples from {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return data


def export_gestures(
    input_dir: str = 'data/raw',
    output_file: str = 'gestures.json',
    samples_per_gesture: int = 5,
    include_all: bool = False
) -> Dict[str, Any]:
    """
    Export gesture examples to JSON file.
    
    Args:
        input_dir: Directory with CSV data
        output_file: Output JSON path
        samples_per_gesture: Max examples per gesture (0 or negative for all)
        include_all: Include all samples if True
        
    Returns:
        Exported gestures dictionary
    """
    print("\n" + "=" * 50)
    print("EXPORT GESTURES TO JSON")
    print("=" * 50)
    
    # Load existing gestures.json if exists
    existing = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing = json.load(f)
            print(f"Loaded existing {output_file} with {len(existing)} gestures")
        except:
            pass
    
    # Load CSV data
    csv_data = load_csv_data(input_dir)
    
    if not csv_data:
        print("No data to export!")
        return existing
    
    # Merge with existing
    result = {}
    
    all_gestures = set(existing.keys()) | set(csv_data.keys())
    
    for gesture in sorted(all_gestures):
        # Get description from existing or create default
        if gesture in existing:
            desc = existing[gesture].get('description', f'{gesture} gesture')
        else:
            desc = f'{gesture} gesture'
        
        # Get examples
        examples = []
        
        if gesture in csv_data:
            samples = csv_data[gesture]
            
            if include_all or samples_per_gesture <= 0:
                examples = samples
            else:
                # Random sample if more than requested
                if len(samples) > samples_per_gesture:
                    indices = np.random.choice(len(samples), samples_per_gesture, replace=False)
                    examples = [samples[i] for i in indices]
                else:
                    examples = samples
        
        result[gesture] = {
            'description': desc,
            'examples': examples
        }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Summary
    print("\n" + "-" * 50)
    print("EXPORT SUMMARY")
    print("-" * 50)
    
    total_examples = 0
    for gesture, data in result.items():
        n_examples = len(data['examples'])
        total_examples += n_examples
        print(f"  {gesture}: {n_examples} examples")
    
    print("-" * 50)
    print(f"Total: {len(result)} gestures, {total_examples} examples")
    print(f"Saved to: {output_file}")
    print("=" * 50 + "\n")
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Export gestures to JSON')
    parser.add_argument('--input', '-i', type=str, default='data/raw',
                       help='Input directory with CSV files')
    parser.add_argument('--output', '-o', type=str, default='gestures.json',
                       help='Output JSON file')
    parser.add_argument('--samples', '-s', type=int, default=5,
                       help='Max samples per gesture (0 for all)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Include all samples')
    
    args = parser.parse_args()
    
    export_gestures(
        input_dir=args.input,
        output_file=args.output,
        samples_per_gesture=args.samples,
        include_all=args.all
    )


if __name__ == '__main__':
    main()
