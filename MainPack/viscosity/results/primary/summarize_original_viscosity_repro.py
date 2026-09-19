from scipy.io import loadmat
import numpy as np
from pathlib import Path

mat_path = Path(__file__).resolve().with_name("viscosity_results.mat")
m = loadmat(mat_path)

print("MAT file:", mat_path)
print("Keys:", ", ".join(k for k in m.keys() if not k.startswith("__")))

for k, v in m.items():
    if k.startswith("__"):
        continue
    arr = np.asarray(v).squeeze()
    if np.issubdtype(arr.dtype, np.number):
        if arr.size <= 20:
            print(f"{k} = {arr}")
        else:
            flat = arr.flatten()
            print(f"{k}: shape={arr.shape}, first_values={flat[:5]}")
