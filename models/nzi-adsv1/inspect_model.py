#!/usr/bin/env python3
"""
Inspect a PyTorch .pt file to extract version information and metadata
without fully loading the model.
"""
import pickle
import sys
from pathlib import Path
import pathlib
import platform

# Fix Windows path issue
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

def inspect_pt_file(pt_path):
    """Inspect a .pt file and extract metadata."""
    print(f"Inspecting: {pt_path}\n")
    
    with open(pt_path, 'rb') as f:
        # Try to peek at the pickle without fully unpickling
        try:
            unpickler = pickle.Unpickler(f)
            
            # Get the first object (usually metadata or the model dict)
            obj = unpickler.load()
            
            print("=" * 60)
            print("TOP-LEVEL OBJECT TYPE")
            print("=" * 60)
            print(f"Type: {type(obj)}\n")
            
            # If it's a dict, show keys
            if isinstance(obj, dict):
                print("=" * 60)
                print("DICTIONARY KEYS")
                print("=" * 60)
                for key in obj.keys():
                    print(f"  - {key}")
                print()
                
                # Check for common version keys
                version_keys = ['ultralytics_version', 'version', 'pytorch_version', 
                               'python_version', 'train_args', 'model_info', 'metadata']
                
                print("=" * 60)
                print("VERSION INFORMATION")
                print("=" * 60)
                found_version = False
                for key in version_keys:
                    if key in obj:
                        print(f"{key}: {obj[key]}")
                        found_version = True
                
                if not found_version:
                    print("No explicit version keys found")
                print()
                
                # Check for train_args or similar
                if 'train_args' in obj:
                    print("=" * 60)
                    print("TRAINING ARGUMENTS")
                    print("=" * 60)
                    print(obj['train_args'])
                    print()
                
                # Check model architecture info
                if 'model' in obj:
                    print("=" * 60)
                    print("MODEL INFO")
                    print("=" * 60)
                    model_obj = obj['model']
                    print(f"Model type: {type(model_obj)}")
                    if hasattr(model_obj, '__dict__'):
                        print(f"Model attributes: {list(model_obj.__dict__.keys())[:10]}")
                    print()
            
            # Try to get module references
            print("=" * 60)
            print("MODULE REFERENCES (from pickle)")
            print("=" * 60)
            f.seek(0)
            content = f.read()
            
            # Look for ultralytics references
            if b'ultralytics' in content:
                print("✓ Contains 'ultralytics' references")
                # Try to find version strings
                if b'8.0' in content:
                    print("  - Found '8.0' in file")
                if b'8.1' in content:
                    print("  - Found '8.1' in file")
                if b'8.2' in content:
                    print("  - Found '8.2' in file")
                if b'8.3' in content:
                    print("  - Found '8.3' in file")
                if b'8.4' in content:
                    print("  - Found '8.4' in file")
            
            # Look for specific module paths
            modules = [b'ultralytics.utils', b'ultralytics.nn', b'ultralytics.models']
            for module in modules:
                if module in content:
                    print(f"✓ References: {module.decode()}")
            
        except Exception as e:
            print(f"Error during inspection: {e}")
            print("\nTrying alternative method...")
            
            # Try torch.load with weights_only=False
            import torch
            try:
                f.seek(0)
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
                print("\n" + "=" * 60)
                print("TORCH.LOAD SUCCESSFUL")
                print("=" * 60)
                print(f"Checkpoint type: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"Keys: {list(checkpoint.keys())}")
            except Exception as e2:
                print(f"Torch load also failed: {e2}")

if __name__ == "__main__":
    # Download the model first
    from huggingface_hub import hf_hub_download
    
    if len(sys.argv) > 1:
        pt_file = sys.argv[1]
    else:
        print("Downloading model from HuggingFace...")
        pt_file = hf_hub_download(
            repo_id='Addax-Data-Science/NZI-ADS-v1',
            filename='new_zealand_v1.pt'
        )
        print(f"Downloaded to: {pt_file}\n")
    
    inspect_pt_file(pt_file)
