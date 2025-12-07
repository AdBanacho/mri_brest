import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm


def load_or_compute_sizes(df, size_cache_path, image_root, serie_col):
    if os.path.exists(size_cache_path):
        size_df = pd.read_csv(size_cache_path)

        size_map = {
            row["serie"]: (int(row["D"]), int(row["H"]), int(row["W"]))
            for _, row in size_df.iterrows()
        }

        sizes = []
        missing = []
        for serie in df[serie_col]:
            if serie in size_map:
                sizes.append(size_map[serie])
            else:
                missing.append(serie)

        if missing:
            new_records = []
            for serie in tqdm(missing, desc="Computing missing shapes"):
                path = os.path.join(image_root, f"{serie}.nii.gz")
                img = nib.load(path)
                vol = np.squeeze(img.get_fdata().astype(np.float32))
                D, H, W = vol.shape
                size_map[serie] = (D, H, W)
                new_records.append({"serie": serie, "D": D, "H": H, "W": W})

            if new_records:
                size_df = pd.concat(
                    [size_df, pd.DataFrame(new_records)], ignore_index=True
                )
                size_df.to_csv(size_cache_path, index=False)

            sizes = [size_map[serie] for serie in df[serie_col]]

        return sizes

    records = []
    sizes = []

    for i in tqdm(range(len(df)), desc="Computing shapes"):
        serie = df.iloc[i][serie_col]
        path = os.path.join(image_root, f"{serie}.nii.gz")

        img = nib.load(path)
        vol = np.squeeze(img.get_fdata().astype(np.float32))
        D, H, W = vol.shape

        records.append({"serie": serie, "D": D, "H": H, "W": W})
        sizes.append((D, H, W))

    size_df = pd.DataFrame(records)
    size_df.to_csv(size_cache_path, index=False)

    return sizes