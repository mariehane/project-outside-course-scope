#%%
import sys
import os
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/Caskroom/slicer/4.11.20210226,1442768/Slicer.app/Contents/MacOS:" + os.environ["PATH"]
os.environ["SITK_SHOW_COMMAND"] = "Slicer"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pydicom import dcmread
from tqdm import tqdm

#%%
#raise KeyboardInterrupt # fix bug with vscode-jupyter where it tries to run all the code on startup

#%%
reader = sitk.ImageSeriesReader()

dicom_path = Path("/Users/mariehane/Desktop/LIDC-IDRI/manifest-1600709154662/LIDC-IDRI")

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
#sitk.Show(image, "Dicom Series", debugOn=True)

# using GetDCMSeriesFileNames
#first_scan_path = Path(first_scan["path"].iloc[0]).parent.parent
#dicom_names = reader.GetGDCMSeriesFileNames(str(first_scan_path), recursive=True)
#print(dicom_names)
#sitk.Show(image, "Dicom Series", debugOn=True)

#%%
# annotate with dicom fields
tqdm.pandas(desc="Reading DCM fields")
def get_dcm_fields(row):
    dcm = dcmread(row["path"], specific_tags=["Modality", "PatientPosition", "ImagePositionPatient"])
    row["Modality"]             = dcm.Modality
    try:
        row["PatientPosition"]      = dcm.PatientPosition
        row["ImagePositionPatientX"] = dcm.ImagePositionPatient[0]
        row["ImagePositionPatientY"] = dcm.ImagePositionPatient[1]
        row["ImagePositionPatientZ"] = dcm.ImagePositionPatient[2]
    except Exception as e:
        row["PatientPosition"]       = np.nan
        row["ImagePositionPatientX"] = np.nan
        row["ImagePositionPatientY"] = np.nan
        row["ImagePositionPatientZ"] = np.nan
        #print(row)
        #print(e)
    return row

if Path("df.csv").exists():
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
#sitk.Show(image, "Dicom series")

#%%
# check if there are different sizes
scan_ids = df["Scan ID"].unique()

x = None
if Path("sizes_spacings.csv").exists():
    x = pd.read_csv("sizes_spacings.csv")
else:
    def get_size_spacing(scan_id):
        paths = df[df["Scan ID"] == scan_id]["path"]
        reader.SetFileNames(tuple(paths))
        image = reader.Execute()
        size = image.GetSize()
        spacing = image.GetSpacing()

        return {
            "scan_id": scan_id,
            "Size X": size[0],
            "Size Y": size[1],
            "Size Z": size[2],
            "Spacing X": size[0],
            "Spacing Y": spacing[1],
            "Spacing Z": spacing[2]
        }

        #tqdm.write(f"Scan {scan_id} has size {size} and spacing {spacing}")
    x = pd.DataFrame(tqdm(map(get_size_spacing, scan_ids), total=len(scan_ids)))
    x.to_csv("sizes_spacings.csv")

x.describe()

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

#sitk.Show(image)
#%% - utility funcs

def load_image(scan_id):
    scan = df[df["Scan ID"] == scan_id]
    scan.sort_values(by="ImagePositionPatientZ", inplace=True)
    paths = scan["path"]
    reader.SetFileNames(tuple(paths))
    image = reader.Execute()
    return image

def view_slice(image, i=50, plane="Axial", cmap='gray'):
    if type(image) == sitk.SimpleITK.Image:
        array = sitk.GetArrayViewFromImage(image)
    else:
        array = image

    if plane == "Axial":
        plt.imshow(array[i,:,:], cmap=cmap)
    elif plane == "Coronal":
        plt.imshow(array[:,i,:], cmap=cmap)
    elif plane == "Sagittal":
        plt.imshow(array[:,:,i], cmap=cmap)

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

# effect is really visible coronal plane
view_slice(img_small, i=180, plane="Coronal")
plt.title("Before resampling to isotropic spacing")
plt.show()
view_slice(img_small_iso, i=180, plane="Coronal")
plt.title("After resampling to isotropic spacing")
plt.show()

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

#resampler_iso_crop = sitk.ResampleImageFilter()
#resampler_iso_crop.SetInterpolator(sitk.sitkLinear)
#resampler_iso_crop.SetOutputSpacing((1,1,1))
#
#size_small = img_small.GetSize()
#spacing_small = img_small.GetSpacing()
#min_size = (int(size_small[0] * spacing_small[0]),
#            int(size_small[1] * spacing_small[1]),
#            int(size_small[2] * spacing_small[2]))
#resampler_iso_crop.SetSize(min_size) 
#
#def resample_iso_crop(img):
#    """Resamples the given image to isotropic spacing and crops it to the smallest image size (centered around the middle)
#    """
#    resampler_iso_crop.SetOutputDirection(img.GetDirection())
#    resampler_iso_crop.SetOutputOrigin(img.GetOrigin())
#
#    size = img.GetSize()
#    offset = (np.array(size) - np.array(min_size))/2
#    transform = sitk.TranslationTransform(3, offset) # XXX: should maybe be reversed
#    resampler_iso_crop.SetTransform(transform)
#
#    return resampler_iso_crop.Execute(img)

# %% view 

images = map(load_image, scan_ids)
images_iso = map(resample_iso, images)
arrays = map(sitk.GetArrayFromImage, images_iso)

img1 = next(images_iso)
print(img1.GetSize())

array1 = next(arrays)
print(array1)
print(array1.shape)

# %%
import torch
from data import LIDCIDRIDataset
dataset = LIDCIDRIDataset(path="/Users/mariehane/Desktop/LIDC-IDRI")

loader = torch.utils.data.DataLoader(dataset)
#sizes = list(tqdm(map(lambda t: t.shape, loader)))

#crop = torchio.transforms.CropOrPad(min_size)
#transforms = torch.nn.Sequential(
#    transforms.CenterCrop(10),
#    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#)

#%%
# find smallest tensor dimensions:
#shapes = []
#for x1 in loader:
#    shapes.append(x1.shape)
#    print(x1.shape)
#shapes = torch.tensor(shapes)
#print(shapes.min(dim=0))

#%% 
# see if theres 100% overlap between diagnoses and CT scans
patient_ids = set(df["Patient ID"].unique())
print("There are", len(patient_ids), "patient IDs in total")

diagnosis_ids = dataset.diagnosis_data["TCIA Patient ID"].unique()
print("There are", len(diagnosis_ids), "patient IDs in the diagnosis data")

n_overlap = len(patient_ids.intersection(diagnosis_ids))
print("There are", n_overlap, "which are in both sets")

#%%
# view distribution of no. of images
n_images = df.groupby("Scan ID").count().iloc[:,0]
n_images = n_images.sort_values()
print(n_images)
plt.hist(n_images)
plt.show()

print("The patients with the fewest images are:")
for i in range(5):
    scan_id = n_images.keys()[i]
    print(df[df["Scan ID"] == scan_id]["Patient ID"].iloc[0], f"({n_images[scan_id]} images)")

#%%
# pylidc - the official LIDC-IDRI python package
#!pip install pylidc
with open("/Users/mariehane/.pylidcrc", "w") as f:
    f.write(f"""[dicom]
    path = {dicom_path}
    warn = True
    """)
import pylidc

scans = pylidc.query(pylidc.Scan).all()

# get patients with malignant tumors
malignant_ids = dataset.diagnosis_data[dataset.diagnosis_data.iloc[:,1] == 2]["TCIA Patient ID"]
patient_id = malignant_ids.iloc[0]
scan = pylidc.query(pylidc.Scan).filter(pylidc.Scan.patient_id == patient_id).first()

nods = scan.cluster_annotations()
scan.visualize(annotation_groups=nods)

# view all annotations
annotations = pylidc.query(pylidc.Annotation).filter(pylidc.Annotation.scan_id == scan.id).all()
for ann in annotations:
    ann.visualize_in_scan()

#%%
# simple segmentation with connected-components-3d
#!pip install connected-components-3d
import cc3d
ref_img = dataset._load_image("01-01-2000-CT CHEST W CONT-45499")
ref_img_iso = resample_iso(ref_img)
array = sitk.GetArrayFromImage(ref_img)

labels_out = cc3d.connected_components(array, connectivity=26) # 26 is the default connectivity
# cc3d.each is VERY slow
#for label, image in cc3d.each(labels_out, binary=False, in_place=True):
    #print(image)
    #view_slice(image)
    #break
view_slice(labels_out, cmap=None)

#%%
# segmentation with SimpleITK
ref_img = dataset._load_image("01-01-2000-CT CHEST W CONT-45499")
ref_img = dataset._resample_iso(ref_img)

# region growing segmentation
seed = (116,177,277) # TODO: find proper way to seed
seg = sitk.ConfidenceConnected(ref_img, seedList=[seed],
                               numberOfIterations=1,
                               multiplier=2.5,
                               initialNeighborhoodRadius=1,
                               replaceValue=1)

sitk.Show(sitk.LabelOverlay(ref_img, seg), "ConfidenceConnected")

#%% segment out air
size = ref_img.GetSize()
topandbottom = [
    (int(size[0]/2), 1,              int(size[2]/2)),
    (int(size[0]/2), int(size[1])-1, int(size[2]/2)),
]

air_seg = sitk.ConfidenceConnected(ref_img, seedList=topandbottom,
                               numberOfIterations=1,
                               multiplier=2.5,
                               initialNeighborhoodRadius=1,
                               replaceValue=1)

sitk.Show(sitk.LabelOverlay(ref_img, seg), "ConfidenceConnected")

#%% threshold segmentation
# view histogram
#plt.hist(sitk.GetArrayViewFromImage(ref_img).flatten())

# simple threshold based on histogram
seg = sitk.BinaryThreshold(ref_img, lowerThreshold=-800, upperThreshold=3000, insideValue=0, outsideValue=1)

sitk.Show(sitk.LabelOverlay(ref_img, seg), "Otsu Thresholding")


#%%
# threshold all air (lungs and outside)
air_seg = sitk.BinaryThreshold(ref_img, lowerThreshold=-800, upperThreshold=3000, insideValue=0, outsideValue=1)

# remove air on the outside
size = air_seg.GetSize()
topandbottom = [
    (int(size[0]/2), 1,              int(size[2]/2)),
    (int(size[0]/2), int(size[1])-1, int(size[2]/2)),
]

outside_seg = sitk.ConfidenceConnected(air_seg, seedList=topandbottom,
                               numberOfIterations=1,
                               multiplier=2.5,
                               initialNeighborhoodRadius=1,
                               replaceValue=1)


# remove outside air
inside_seg = air_seg - outside_seg

# XXX: at this point maybe clean a bit with opening/closing

# compute different segments using watershedding
dist_img = sitk.SignedMaurerDistanceMap(inside_seg != 0, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
#sitk.Show(dist_img, "Distance")

# Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
radius = 10
seeds = sitk.ConnectedComponent(dist_img < radius)
# Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
ws = sitk.Mask(ws, sitk.Cast(inside_seg, ws.GetPixelID()))

# choose the two largest segments
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(ws)

sizes = dict((label, shape_stats.GetPhysicalSize(label)) for label in shape_stats.GetLabels())
largest_sizes_idx = np.argsort(list(sizes.values()))[::-1]

# if 2nd largest segment is less than 1/32 the size of the largest then ignore, otherwise combine them
# TODO: determine if this any lung cts will have two segments that are joined with this procedure
labels = np.array(list(sizes.keys()))
labels_sorted = labels[largest_sizes_idx]

lungs = ws == labels_sorted[0]
if sizes[labels_sorted[1]] > sizes[labels_sorted[0]]/32:
    print("Combining with ")
    lungs |= ws == labels_sorted[1]

sitk.Show(sitk.LabelOverlay(ref_img, lungs), "Lungs")

#%% 
# Previous cell, but as a function
def segment_lungs(img):
    air_seg = sitk.BinaryThreshold(img, lowerThreshold=-800, upperThreshold=3000, insideValue=0, outsideValue=1)

    # remove air on the outside
    size = air_seg.GetSize()
    topandbottom = [
        (int(size[0]/2), 1,              int(size[2]/2)),
        (int(size[0]/2), int(size[1])-1, int(size[2]/2)),
    ]

    outside_seg = sitk.ConfidenceConnected(air_seg, seedList=topandbottom,
                                numberOfIterations=1,
                                multiplier=2.5,
                                initialNeighborhoodRadius=1,
                                replaceValue=1)


    # remove outside air
    inside_seg = air_seg - outside_seg

    # XXX: at this point maybe clean a bit with opening/closing

    # compute different segments using watershedding
    dist_img = sitk.SignedMaurerDistanceMap(inside_seg != 0, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
    #sitk.Show(dist_img, "Distance")

    # Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
    radius = 10
    seeds = sitk.ConnectedComponent(dist_img < radius)
    # Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
    seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
    ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
    ws = sitk.Mask(ws, sitk.Cast(inside_seg, ws.GetPixelID()))

    # choose the two largest segments
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(ws)

    sizes = dict((label, shape_stats.GetPhysicalSize(label)) for label in shape_stats.GetLabels())
    largest_sizes_idx = np.argsort(list(sizes.values()))[::-1]

    # if 2nd largest segment is less than 1/32 the size of the largest then ignore, otherwise combine them
    # TODO: determine if this any lung cts will have two segments that are joined with this procedure
    labels = np.array(list(sizes.keys()))
    labels_sorted = labels[largest_sizes_idx]

    lungs = ws == labels_sorted[0]
    if sizes[labels_sorted[1]] > sizes[labels_sorted[0]]/32:
        print("Combining largest segment with 2nd largest segment")
        lungs |= ws == labels_sorted[1]

    return lungs

#%%
# TODO: Registration
def get_scans(patient_id):
    images = df[df["Patient ID"] == patient_id]
    scan_ids = images["Scan ID"].unique()
    return scan_ids

# this is just one I picked because it was diagnosed as benign in the diagnosis spreadsheet and I found it reasonably normal (couldn't find a nodule, and the lungs seemed large and centered)
ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
# one with a visible nodule, (in the top of the left lung)
img = dataset._load_image("01-01-2000-CT CHEST W CONT-45499")

# compute region of interest (lungs)
ref_lungs = segment_lungs(ref_img)
img_lungs = segment_lungs(img)


# perform the actual registration

initial_transform = sitk.CenteredTransformInitializer(ref_lungs, 
                                                      img_lungs, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
#img_resampled = sitk.Resample(img, ref_img, initial_transform, sitk.sitkLinear, 0.0, img.GetPixelID())

registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()

registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

registration_method.SetInitialTransform(initial_transform, inPlace=False)

transform = registration_method.Execute(sitk.Cast(ref_lungs, sitk.sitkFloat32), 
                                               sitk.Cast(img_lungs, sitk.sitkFloat32))

img_transformed = sitk.Resample(img, ref_lungs, transform, sitk.sitkLinear, 0.0, img.GetPixelID())

# inspect results

view_slice(dataset._resample_iso(img), i=180)
plt.title("Image before registration")
plt.show()
view_slice(dataset._resample_iso(ref_img), i=180)
plt.title("Reference Image")
plt.show()
view_slice(dataset._resample_iso(img_transformed), i=180)
plt.title("Image after registration")
plt.show()

#img_lungs_transformed = sitk.Resample(img_lungs, ref_lungs, final_transform, sitk.sitkLinear, 0.0, img_lungs.GetPixelID())
#view_slice(dataset._resample_iso(img_lungs), i=180)
#plt.title("Lung segmentation")
#plt.show()
#view_slice(dataset._resample_iso(ref_lungs), i=180)
#plt.title("Reference lung segmentation")
#plt.show()
#view_slice(dataset._resample_iso(img_lungs_transformed), i=180)
#plt.title("Lung segmentation after registration")
#plt.show()

#%%
# prev. cell as a function
def get_registration_transform(img_lungs, ref_lungs):
    initial_transform = sitk.CenteredTransformInitializer(ref_lungs, 
                                                        img_lungs, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.01)

    method.SetInterpolator(sitk.sitkLinear)

    method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    method.SetOptimizerScalesFromPhysicalShift()

    method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = method.Execute(sitk.Cast(ref_lungs, sitk.sitkFloat32), 
                                                sitk.Cast(img_lungs, sitk.sitkFloat32))
    return final_transform

def register(img, ref_lungs, transform):
    return sitk.Resample(img, ref_lungs, transform, sitk.sitkLinear, 0.0, img.GetPixelID())

#%% use registration to detect nodule
#ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
#img = dataset._load_image("01-01-2000-CT CHEST W CONT-45499")
#img_lungs = segment_lungs(img)
#ref_lungs = segment_lungs(ref_img)
final_transform = get_registration_transform(img_lungs, ref_lungs)
lungs_registered = register(img_lungs, ref_lungs, final_transform)

#diff1 = ref_lungs - lungs_registered
#diff2 = lungs_registered - ref_lungs
#diff3 = ref_lungs & lungs_registered
#sitk.Show(diff1, "Diff1")
#sitk.Show(diff2, "Diff2")
#sitk.Show(diff3, "Diff3")

# %% 
# ML Preprocessing Pipeline
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from torchio.transforms import CropOrPad

from data import LIDCIDRIDataset, Register, RegisterUsingLungSegmentation, ResampleIsotropic, SITKImageToTensor

dataset = LIDCIDRIDataset(path="/Users/mariehane/Desktop/LIDC-IDRI")
ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
translate = sitk.TranslationTransform(3)
translate.SetParameters((0, 40, 0)) # slight adjustment to put it closer to the center
ref_img = sitk.Resample(ref_img, translate)

dataset.transform = Compose([
        Register(ref_img),
        #RegisterUsingLungSegmentation(ref_img), # slow
        ResampleIsotropic(),
        SITKImageToTensor(),
        CropOrPad(target_shape=(256, 256, 256))
    ])

# view the first 5
#for i, (img, label) in enumerate(dataset):
#    if i >= 5:
#        break
#
#    print("Label:", label)
#    print("Tensor shape:", img.shape)
#    #print("Tensor:")
#    #print(img)
#
#    plt.imshow(img[0,180,:,:], cmap='gray')
#    plt.show()

# Super basic CNN using PyTorch lightning
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

class LungCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential( # inputs: 256x256x256
             nn.Conv3d(in_channels=1, out_channels=1, kernel_size=5, stride=5), # -> 51x51x51
             nn.ReLU(),
             nn.MaxPool3d(kernel_size=20), # 32x32x32
             nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, dilation=3), # -> 9x9x9
             nn.ReLU(),
             nn.MaxPool3d(kernel_size=5), # -> 4x4x4
             nn.ReLU(),
             nn.Flatten(start_dim=3, end_dim=1), # flatten to 64
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax()
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.cnn(x)
        out = self.dense(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        #x = x.view(x.size(0), -1)
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


loader = DataLoader(dataset)
classifier = LungCNN()
trainer = pl.Trainer() # num_processes=X, gpus=Y, ...
trainer.fit(classifier, loader)

# %%
# TODO: View histograms of intensities to see if you need to rescale
# TODO: view dist. of sizes (or no. of images) to see if theres a pattern to which scans to remove
