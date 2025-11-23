import os
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

base_path = "tciaDownload"

folder_list = []

for item in os.listdir(base_path):
    full_path = os.path.join(base_path, item)
    if os.path.isdir(full_path):
        folder_list.append(full_path)


dcm_files = []

for folder in folder_list:
    listdir = os.listdir(folder)
    half = len(listdir) // 2
    for inx, file in enumerate(listdir):
        if file.lower().endswith(".dcm") and half == inx:
            dcm_files.append(os.path.join(folder, file))
            break

results = []

for dcm_path in tqdm(dcm_files):
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=False)

        info = {
            "file": dcm_path,
            # "Institution Name": ds.get((0x0008, 0x0080), "").value if ds.get((0x0008, 0x0080)) else None,
            # "Institution Address": ds.get((0x0008, 0x0081), "").value if ds.get((0x0008, 0x0081)) else None,
            # "Institutional Department Name": ds.get((0x0008, 0x1040), "").value if ds.get((0x0008, 0x1040)) else None,
            # "Station Name": ds.get((0x0008, 0x1010), "").value if ds.get((0x0008, 0x1010)) else None,
            "Manufacturer Model Name": ds.get((0x0008, 0x1090), "").value if ds.get((0x0008, 0x1090)) else None,
            # "Clinical Trial Coordinating Center": ds.get((0x0012, 0x0060), "").value if ds.get((0x0012, 0x0060)) else None,
        }

        results.append(info)

    except Exception as e:
        print(f"Error reading {dcm_path}: {e}")

unique_models = {}
for item in results:
    model = item["Manufacturer Model Name"]
    if model not in unique_models:
        unique_models[model] = item["file"]
    elif model + " 2" not in unique_models:
        unique_models[model+" 2"] = item["file"]

fig, axes = plt.subplots(2, 7, figsize=(21, 6))

axes = axes.flatten()

for ax, (model, dcm_path) in zip(axes, unique_models.items()):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    ax.imshow(img)
    ax.set_title(model)
    ax.axis("off")

plt.tight_layout()
plt.show()

models = [item["Manufacturer Model Name"] for item in results]

model_counts = Counter(models)

labels = list(model_counts.keys())
values = list(model_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel("Manufacturer Model Name")
plt.ylabel("Count")
plt.title("Distribution of DICOM Scanner Models")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()