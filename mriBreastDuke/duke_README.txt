 3D Breast, FGT, and Blood Vessel MRI Segmentation Supplemental Data

This folder contains supplemental data used for the paper titled "Automatic Breast, Fibroglandular Tissue, and Blood Vessel Segmentation in Breast MRI using Convolutional Neural Networks". The objective of this study was to develop and validate a fully automated deep-learning segmentation method based on convolutional neural network (CNN) for breast, fibroglandular tissue (FGT, also referred to as dense tissue), and blood vessel segmentation as well as to publicly share annotation masks and all code used in the study. This folder contains the data used in the study in a format to supplement the data contained within the TCIA database. 

The image slices were taken from fat-saturated gradient echo T1-weighted pre-contrast MRI studies in the Duke Breast Cancer MRI dataset (https://doi.org/10.7937/TCIA.e3sv-re93). We randomly selected 100 subjects to be included in our study. 70 subjects were used in the training set, 15 in the validation set, and 15 in the testing set. For each subject, the breast tissue, FGT, and blood vessels were manually annotated using 3D Slicer according to the following rules: 
	1.	Breast tissue annotation: Trace the contours of the breast, excluding inner anatomical structures, such as the chest wall muscles and sternum. The superior edge of the annotation should stop approximately at the level of the clavicle. The inferior edge of the annotation should stop at the inframammary fold. 
	2.	FGT annotation: Mark all areas that appear to represent fibroglandular tissue that do not appear to be blood vessels. Biopsy clips and lymph nodes should be excluded. All FGT should be confined to within the breast. 
	3.	Blood vessel annotation: Mark all readily apparent blood vessels. All blood vessels should be confined to within the breast.
 The FGT and blood vessel annotations are stored within the same file as they are non-overlapping. Both of these annotations are contained within the breast annotation that is stored in a separate file. All annotations were reviewed by board-certified, fellowship-trained breast radiologists. This dataset contains all data used in the study.

The segmentation masks within this folder are stored in the NRRD file format. Each pixel in the image array is either a 0 or 1, to indicate an absence of or presence of the target tissue. 

The segmentation mask files in this supplemental folder have the following naming scheme:

Breast_MRI_{patient_ID}_{segmentation_tissue_target}.seg.nrrd

# File Directory

.
├── Segmentation_Masks_NRRD       		           					# Directory containing true segmentation masks
│   ├── Breast_MRI_002       										# Subject directory containing breast, FGT, and blood vessel mask
│   |   ├── Segmentation_Breast_MRI_002_Breast.seg.nrrd 			# Breast mask for subject Breast_MRI_002
│   |   ├── Segmentation_Breast_MRI_002_Dense_and_Vessels.seg.nrrd  # FGT and blood vessel mask for subject Breast_MRI_002
│   ...