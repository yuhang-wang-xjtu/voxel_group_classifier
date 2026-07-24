"""
Colab Cell 3: Download S3DIS HDF5 from HuggingFace and convert to Pointcept format.
Runs in system Python (no conda needed for data download).
"""
import numpy as np, h5py, os, glob, urllib.request, zipfile, sys

HDF5_URL = "https://huggingface.co/datasets/cminst/S3DIS/resolve/main/indoor3d_sem_seg_hdf5_data.zip"
HDF5_DIR = "./indoor3d_sem_seg_hdf5_data"
OUT_ROOT = "./data/s3dis"

# Download HDF5
if not os.path.isdir(HDF5_DIR):
    print("Downloading S3DIS HDF5 from HuggingFace (~1.7 GB)...")
    zip_path = "/tmp/s3dis.zip"
    urllib.request.urlretrieve(HDF5_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(HDF5_DIR)
    os.remove(zip_path)
else:
    print("S3DIS data already downloaded.")

# Find HDF5 files
h5_files = sorted(glob.glob(f"{HDF5_DIR}/**/*.h5", recursive=True))
if not h5_files:
    h5_files = sorted(glob.glob(f"{HDF5_DIR}/*.h5"))
print(f"Found {len(h5_files)} HDF5 files.")

# Convert each HDF5 file → Pointcept format
for h5_path in h5_files:
    fname = os.path.basename(h5_path).replace(".h5", "")
    area_idx = int(fname.split("_")[-1])

    # Only use HDF5 file 0 → Area_1, rest skip (smoke test)
    if area_idx != 0:
        continue

    area = "Area_1"
    # Use the HDF5 filename as room name
    room_dir = os.path.join(OUT_ROOT, area, fname)
    if os.path.exists(os.path.join(room_dir, "normal.npy")):
        continue  # already converted

    os.makedirs(room_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]
        labels = f["label"][:]

    np.save(os.path.join(room_dir, "coord.npy"), data[..., :3].reshape(-1, 3).astype(np.float32))
    np.save(os.path.join(room_dir, "color.npy"), data[..., 3:6].reshape(-1, 3).astype(np.uint8))
    np.save(os.path.join(room_dir, "normal.npy"), data[..., 6:9].reshape(-1, 3).astype(np.float32))
    np.save(os.path.join(room_dir, "segment.npy"), labels.reshape(-1, 1).astype(np.int16))
    np.save(os.path.join(room_dir, "instance.npy"), np.zeros_like(labels.reshape(-1, 1), dtype=np.int16))

# Verify
area_dirs = sorted(glob.glob(f"{OUT_ROOT}/Area_*"))
total_rooms = sum(len(os.listdir(a)) for a in area_dirs)
print(f"Done. {total_rooms} rooms across {len(area_dirs)} areas.")
for a in area_dirs:
    print(f"  {os.path.basename(a)}: {len(os.listdir(a))} rooms")
