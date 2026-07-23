# HDF5 S3DIS (PointNet) → Pointcept format converter
import numpy as np
import h5py, os, glob

# Use the HDF5 files already downloaded by the smoke test notebook
HDF5_DIR = "./indoor3d_sem_seg_hdf5_data/indoor3d_sem_seg_hdf5_data"
OUT_ROOT = "./data/s3dis"

# Map 13-class S3DIS labels
S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]

h5_files = sorted(glob.glob(f"{HDF5_DIR}/ply_data_all_*.h5"))
print(f"Found {len(h5_files)} HDF5 files")

for h5_path in h5_files:
    fname = os.path.basename(h5_path).replace(".h5", "")  # ply_data_all_0
    area_idx = int(fname.split("_")[-1])  # 0-23
    
    # Map: 0→Area_1, 1-18→Area_2-5, 19-23→Area_6
    if area_idx == 0:
        area = "Area_1"
    elif 1 <= area_idx <= 4:
        area = f"Area_{area_idx}"
    elif 5 <= area_idx <= 11:
        area = f"Area_{area_idx}"
    elif 12 <= area_idx <= 18:
        area = f"Area_{area_idx}"
    else:
        area = f"Area_6"
    
    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]    # [N_blocks, 4096, 9]
        labels = f["label"][:]  # [N_blocks, 4096, 1]
    
    # Merge all blocks into one room
    xyz = data[..., :3].reshape(-1, 3).astype(np.float32)
    rgb = data[..., 3:6].reshape(-1, 3).astype(np.uint8)
    nrm = data[..., 6:9].reshape(-1, 3).astype(np.float32) if data.shape[-1] >= 9 else None
    seg = labels.reshape(-1, 1).astype(np.int16)
    ins = np.zeros_like(seg, dtype=np.int16)  # no instance labels in HDF5
    
    print(f"  {area}/{fname}: {len(xyz)} points, classes {np.unique(seg)}")
    
    room_dir = os.path.join(OUT_ROOT, area, fname)
    os.makedirs(room_dir, exist_ok=True)
    np.save(os.path.join(room_dir, "coord.npy"), xyz)
    np.save(os.path.join(room_dir, "color.npy"), rgb)
    np.save(os.path.join(room_dir, "segment.npy"), seg)
    np.save(os.path.join(room_dir, "instance.npy"), ins)
    if nrm is not None:
        np.save(os.path.join(room_dir, "normal.npy"), nrm)

print(f"\nDone. Data written to {OUT_ROOT}/")
