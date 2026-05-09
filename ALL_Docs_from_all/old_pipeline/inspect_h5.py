import h5py
import numpy as np

db_path = r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\reference_database_vejle.h5"

with h5py.File(db_path, 'r') as f:
    print("=== H5 Database Structure ===")
    print(f"Keys: {list(f.keys())}")
    print(f"\nAttributes: {dict(f.attrs)}")
    
    for key in f.keys():
        item = f[key]
        print(f"\n{key}:")
        if hasattr(item, 'shape'):
            print(f"  Shape: {item.shape}")
            print(f"  Dtype: {item.dtype}")
            if item.shape[0] > 0:
                print(f"  Sample: {item[0] if len(item.shape) == 1 else item[0][:5]}")
        if hasattr(item, 'attrs'):
            print(f"  Attrs: {dict(item.attrs)}")
