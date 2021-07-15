#%%
import sys
import os
from pathlib import Path

os.environ["SITK_SHOW_COMMAND"] = "Slicer"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pydicom import dcmread
from tqdm import tqdm

#%%
raise KeyboardInterrupt # fix bug with vscode-jupyter where it tries to run all the code on startup

#%%
reader = sitk.ImageSeriesReader()

dicom_path = Path("LIDC-IDRI/manifest-1600709154662/LIDC-IDRI")

first_dicom_path = dicom_path / "LIDC-IDRI-0001/01-01-2000-35511/3000923.000000-62357"
first_dicom_names = reader.GetGDCMSeriesFileNames(str(first_dicom_path.absolute()), recursive=True)
print(first_dicom_names)

def get_dcm_path_parts(path):
    patient_id = path.parts[-4]
    scan_id = path.parts[-3]
    fname = path.name
    return patient_id, scan_id, fname, str(path)
df = pd.DataFrame(map(get_dcm_path_parts, dicom_path.glob("**/*.dcm")), 
                  columns=["Patient ID", "Scan ID", "filename", "path"])

df = df.sort_values(by=["Patient ID"])

#%%
# get filenames for a single scan
first_scan = df[df["Scan ID"] == '01-01-2000-94866']
print(first_scan["Patient ID"])
dicom_names = first_scan["path"]
print(dicom_names)

reader.SetFileNames(tuple(dicom_names))
image = reader.Execute()
size = image.GetSize()
print(size)

#%%
#sitk.WriteImage(image, "test.png")
sitk.Show(image, "Dicom Series", debugOn=True)

# using GetDCMSeriesFileNames
#first_scan_path = Path(first_scan["path"].iloc[0]).parent.parent
#dicom_names = reader.GetGDCMSeriesFileNames(str(first_scan_path), recursive=True)
#print(dicom_names)
#sitk.Show(image, "Dicom Series", debugOn=True)

#%%
# annotate with dicom fields
tqdm.pandas(desc="Reading DCM fields")
def get_dcm_fields(row):
    dcm = dcmread(row["path"])
    row["Modality"]             = dcm.Modality
    try:
        row["PatientPosition"]      = dcm.PatientPosition
        row["ImagePositionPatientX"] = dcm.ImagePositionPatient[0]
        row["ImagePositionPatientY"] = dcm.ImagePositionPatient[1]
        row["ImagePositionPatientZ"] = dcm.ImagePositionPatient[2]
    except Exception as e:
        row["PatientPosition"]      = np.nan
        row["ImagePositionPatientX"] = np.nan
        row["ImagePositionPatientY"] = np.nan
        row["ImagePositionPatientZ"] = np.nan
        #print(row)
        #print(e)
    return row

if Path("df.csv").exists:
    df = pd.read_csv("df.csv")
else:
    df = df.progress_apply(get_dcm_fields, axis=1) # this takes an hour
    df.to_csv("df.csv")

#%%s
# remove DX and CR images
df = df[df["Modality"] == "CT"]

#%%
# fix slice ordering
first_scan = df[df["Scan ID"] == '01-01-2000-94866']
first_scan.sort_values(by="ImagePositionPatientZ", inplace=True)
dicom_names = first_scan["path"]
reader.SetFileNames(tuple(dicom_names))
image = reader.Execute()
sitk.Show(image, "Dicom series")

#%%
# check if there are different sizes
scan_ids = df["Scan ID"].unique()

x = None
if Path("sizes_spacings.csv").exists:
    x = pd.read_csv("sizes_spacings.csv")
else:
    sizes = []
    spacings = []
    for scan_id in tqdm(scan_ids):
        paths = df[df["Scan ID"] == scan_id]["path"]
        reader.SetFileNames(tuple(paths))
        image = reader.Execute()
        size = image.GetSize()
        spacing = image.GetSpacing()

        sizes.append(size)
        spacings.append(spacing)

        #tqdm.write(f"Scan {scan_id} has size {size} and spacing {spacing}")

    sizes_x = list(map(lambda x: x[0], sizes))
    sizes_y = list(map(lambda x: x[1], sizes))
    sizes_z = list(map(lambda x: x[2], sizes))
    spacings_x = list(map(lambda x: x[0], spacings))
    spacings_y = list(map(lambda x: x[1], spacings))
    spacings_z = list(map(lambda x: x[2], spacings))
    x = pd.DataFrame([sizes_x, sizes_y, sizes_z, spacings_x, spacings_y, spacings_z]).T
    x.rename(columns={
        0: "Size X", 
        1: "Size Y",
        2: "Size Z",
        3: "Spacing X",
        4: "Spacing Y",
        5: "Spacing Z"
    }, inplace=True)

x.describe()
x.to_csv("sizes_spacings.csv")

#%%
# find the shortest/smallest scan and view with slicer (not using SimpleITK)
smallest_scan_mask = (x["Size Z"] == 65)
smallest_scan_id = scan_ids[smallest_scan_mask].item()
smallest_scan = df[df["Scan ID"] == smallest_scan_id]
smallest_scan.sort_values(by="ImagePositionPatientZ", inplace=True)
paths = smallest_scan["path"]
reader.SetFileNames(tuple(paths))
image = reader.Execute()
size = image.GetSize()
print(f"The smallest scan is {smallest_scan['Patient ID'].iloc[0]}, {smallest_scan_id} with size {size}")

sitk.Show(image)
#%% - utility funcs

def load_image(scan_id):
    scan = df[df["Scan ID"] == scan_id]
    scan.sort_values(by="ImagePositionPatientZ", inplace=True)
    paths = scan["path"]
    reader.SetFileNames(tuple(paths))
    image = reader.Execute()
    return image

def view_slice(image, i=50):
    array = sitk.GetArrayViewFromImage(image)
    plt.imshow(array[i,:,:], cmap='gray')

img_small = load_image("01-01-2000-82159")
img_first = load_image("01-01-2000-30178")

#%%
# Resampling # 
resample = sitk.ResampleImageFilter()
resample.SetInterpolator(sitk.sitkLinear)
resample.SetOutputDirection(img_small.GetDirection())
resample.SetOutputOrigin(img_small.GetOrigin())
resample.SetOutputSpacing((1,1,1))

# get size of smallest img with isotropic spacing
size = img_small.GetSize()
spacing = img_small.GetSpacing()
size = (int(size[0] * spacing[0]),
        int(size[1] * spacing[1]),
        int(size[2] * spacing[2]))
resample.SetSize(size) 

img_small_iso = resample.Execute(img_small)

view_slice(img_small_iso, i=50)

#%%
resampler_iso = sitk.ResampleImageFilter()
resampler_iso.SetInterpolator(sitk.sitkLinear)
resampler_iso.SetOutputSpacing((1,1,1))

def resample_iso(img):
    """Resamples the given image to isotropic spacing (the sizing will vary between images)
    """
    resampler_iso.SetOutputDirection(img.GetDirection())
    resampler_iso.SetOutputOrigin(img.GetOrigin())
    
    # calclulate sizing to fit all of resampled image
    size = img.GetSize()
    spacing = img.GetSpacing()
    size = (int(size[0] * spacing[0]),
            int(size[1] * spacing[1]),
            int(size[2] * spacing[2]))
    resampler_iso.SetSize(size) 

    return resampler_iso.Execute(img)


resampler_iso_crop = sitk.ResampleImageFilter()
resampler_iso_crop.SetInterpolator(sitk.sitkLinear)
resampler_iso_crop.SetOutputSpacing((1,1,1))

size_small = img_small.GetSize()
spacing_small = img_small.GetSpacing()
min_size = (int(size_small[0] * spacing_small[0]),
            int(size_small[1] * spacing_small[1]),
            int(size_small[2] * spacing_small[2]))
resampler_iso_crop.SetSize(min_size) 

def resample_iso_crop(img):
    """Resamples the given image to isotropic spacing and crops it to the smallest image size (centered around the middle)
    """
    resampler_iso_crop.SetOutputDirection(img.GetDirection())
    resampler_iso_crop.SetOutputOrigin(img.GetOrigin())

    size = img.GetSize()
    offset = (np.array(size) - np.array(min_size))/2
    transform = sitk.TranslationTransform(3, offset) # XXX: should maybe be reversed
    resampler_iso_crop.SetTransform(transform)

    return resampler_iso_crop.Execute(img)

# %% view 

images = map(load_image, scan_ids)
images_iso = map(resample_iso, images)
arrays = map(sitk.GetArrayFromImage, images_iso)

img1 = next(images_iso)
print(img1.GetSize())

array1 = next(arrays)
#print(array1)
#print(array1.shape)

# %%

# get matrices and crop
smallest_size = resample_iso(img_small).GetSize()
smallest_size = smallest_size[::-1] # re-order to fit array dimension order

matrices = map(sitk.GetArrayViewFromImage, images_iso)

def crop_centered(matrix, size=np.array(smallest_size)):
    crop_size = matrix.shape - size

    tl = np.ceil(crop_size / 2)
    br = np.floor(crop_size / 2)

#%%
import torch
import torchvision
import torchio
# pytorch dataset using CenterCrop

class LIDCIDRIDataset(torch.utils.data.IterableDataset):
    def __init__(self, df, scan_ids, reader):
        super(LIDCIDRIDataset).__init__()

        # simply re-use the prev. global vars
        self.metadata = df
        self.scan_ids = scan_ids
        self.reader = reader
    
    def _load_image(self, scan_id):
        scan = self.metadata[self.metadata["Scan ID"] == scan_id]
        scan.sort_values(by="ImagePositionPatientZ", inplace=True)
        paths = scan["path"]
        self.reader.SetFileNames(tuple(paths))
        image = self.reader.Execute()
        return image

    def __iter__(self):
        # TODO: handle multiple workers
        images = map(load_image, self.scan_ids)
        images_iso = map(resample_iso, images)
        arrays = map(sitk.GetArrayFromImage, images_iso)
        tensors = map(torch.tensor, arrays)
        return tensors


# %%
dataset = LIDCIDRIDataset(df, scan_ids, reader)

loader = torch.utils.data.DataLoader(dataset)

crop = torchio.transforms.CropOrPad(min_size)
#transforms = torch.nn.Sequential(
#    transforms.CenterCrop(10),
#    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#)

shapes = []
for x1 in loader:
    shapes.append(x1.shape)
    print(x1.shape)
# %%
