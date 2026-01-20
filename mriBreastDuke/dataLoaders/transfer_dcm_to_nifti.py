import SimpleITK as sitk
import os
from mriBreastDuke.constants import DCM_PATH, NIFTI_PATH

def transfer_dcm_to_nifti():
    folder_list = []
    series = os.listdir(DCM_PATH)
    seriesNifti = os.listdir(NIFTI_PATH)

    for serie in series:
        full_path = os.path.join(DCM_PATH, serie)
        if os.path.isdir(full_path):
            folder_list.append(full_path)

    for folder_path, serie in zip(folder_list, series):
        if serie in seriesNifti:
            print("EXISTS - " + serie)
            continue
        if serie not in folder_path:
            print("XXXXXXXXX NOT EXISTS id: " + serie + " XXXXXXXXX")
        output_nii = NIFTI_PATH + serie + ".nii.gz"
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(folder_path)
        reader.SetFileNames(dicom_series)
        image = reader.Execute()
        sitk.WriteImage(image, output_nii)
        print("Conversion complete! Saved as:", output_nii)


def read_nifti_metadata_images():
    series = os.listdir(DCM_PATH)
    for serie in series:
        nii_path = NIFTI_PATH + serie + ".nii.gz"
        img = sitk.ReadImage(nii_path)

        print("=== Image Information ===")
        print("File:", nii_path)
        print("Dimension:", img.GetDimension())
        print("Size:", img.GetSize())        # number of voxels
        print("Spacing:", img.GetSpacing())  # physical size per voxel
        print("Origin:", img.GetOrigin())    # world coordinate of first voxel
        print("Direction:", img.GetDirection())  # orientation matrix

        print("Pixel Type:", sitk.GetPixelIDValueAsString(img.GetPixelID()))
        print("Number of Components per Pixel:", img.GetNumberOfComponentsPerPixel())

        stats = sitk.StatisticsImageFilter()
        stats.Execute(img)
        print("Min Intensity:", stats.GetMinimum())
        print("Max Intensity:", stats.GetMaximum())
        print("Mean Intensity:", stats.GetMean())


