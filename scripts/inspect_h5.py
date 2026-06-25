"""One-shot: dump structure + attrs of one HDF5 sample."""
import sys, h5py

p = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Administrator\Desktop\test1\h5_output\00000004\20120612T023409_OD.h5"

with h5py.File(p, "r") as f:
    print(f"File: {p}\n")
    print("=== Datasets ===")
    def visit(n, o):
        if isinstance(o, h5py.Dataset):
            print(f"  {n}: shape={o.shape}, dtype={o.dtype}")
    f.visititems(visit)
    print("\n=== Attributes ===")
    for k, v in f.attrs.items():
        s = str(v)
        if len(s) > 150:
            s = s[:150] + "..."
        print(f"  {k} = {s}")
