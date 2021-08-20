from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision
from pydicom import dcmread
import tqdm

class LIDCIDRIDataset(torch.utils.data.IterableDataset):
    """
    Example Usage:
        >>> transform = Compose([
        ...     ResampleIsotropic(),
        ...     SITKImageToTensor()
        ... ])
        >>> dataset = LIDCIDRIDataset(path="LIDC-IDRI", transform=transform)
        >>> img, label, metadata = next(dataset)
        >>> print("Label:", label)
        >>> print(img)
    """
    def __init__(self, path, transform=None):
        super(LIDCIDRIDataset).__init__()

        path = Path(path)
        dicom_path = next(path.glob("manifest-*/LIDC-IDRI"))
        df_path = path / "df.csv" # caching of internally used dataframe

        self.metadata = pd.read_csv(path / "LIDC-IDRI_MetaData.csv")
        self.scans_meta = pd.read_csv(path / "manifest-1600709154662/metadata.csv", on_bad_lines='warn')
        self.diagnosis_data = pd.read_excel(path / "tcia-diagnosis-data-2012-04-20.xls")
        self.transform = transform

        if df_path.exists():
            self._df = pd.read_csv(df_path)
        else:
            def get_dcm_metadata(path):
                patient_id = path.parts[-4]
                scan_id = path.parts[-3]
                fname = path.name

                dcm = dcmread(path)
                modality = dcm.Modality
                imgPosPatientZ = np.nan
                if modality == "CT":
                    imgPosPatientZ = dcm.ImagePositionPatient[2]

                return patient_id, scan_id, modality, imgPosPatientZ, fname, str(path)

            dcm_files = list(dicom_path.glob("**/*.dcm"))
            self._df = pd.DataFrame(tqdm.tqdm(map(get_dcm_metadata, dcm_files), total=len(dcm_files)),
                                    columns=["Patient ID", "Scan ID", "Modality", "ImagePositionPatientZ", "filename", "path"])

            self._df.to_csv(df_path)

        self._reader = sitk.ImageSeriesReader()
        self._resampler = sitk.ResampleImageFilter()
        self._resampler.SetInterpolator(sitk.sitkLinear)
        self._resampler.SetOutputSpacing((1,1,1))

    def get_scans(self, patient_id):
        """Returns the Scan IDs for a given patient"""
        images = self._df[self._df["Patient ID"] == patient_id]
        return images["Scan ID"].unique()

    def _load_image(self, scan_id):
        scan = self._df[self._df["Scan ID"] == scan_id].copy()
        scan.sort_values(by="ImagePositionPatientZ", inplace=True)
        paths = scan["path"]
        self._reader.SetFileNames(tuple(paths))
        image = self._reader.Execute()
        return image

    def __iter__(self):
        """Returns an iterator over the subset of images that also have diagnosis data."""
        # TODO: handle multiple workers

        # select only patients which are also in the diagnosis data spreadsheet
        patient_ids = self.diagnosis_data.iloc[:, 0].unique()
        df_subset = self._df[self._df["Patient ID"].isin(patient_ids)]

        # drop DX/CR scans
        df_subset = df_subset[df_subset["Modality"] == "CT"]

        scan_ids = df_subset["Scan ID"].unique()

        for scan_id in scan_ids:
            scan_df = df_subset[df_subset["Scan ID"] == scan_id]
            patient_id = scan_df["Patient ID"].iat[0]
            patient_diagnosis_data = self.diagnosis_data[self.diagnosis_data["TCIA Patient ID"] == patient_id]
            target = patient_diagnosis_data.iloc[0,1] # "Diagnosis at the patient level"

            metadata = {
                "patient_id": patient_id,
                "scan_id": scan_id
            }

            img = self._load_image(scan_id)

            try:
                if self.transform:
                    img = self.transform(img)
            except Exception as e:
                print(e)
                from pprint import pprint; pprint(metadata)
                import sys; sys.exit(1)

            yield img, target, metadata

    def __len__(self):
        return len(self.diagnosis_data)

# transforms:
def get_registration_transform(image, reference):
    # SITK's registration framework only supports float32 or float64 pixel types
    image = sitk.Cast(image, sitk.sitkFloat32)
    reference = sitk.Cast(reference, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(reference,
                                                          image,
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

    final_transform = method.Execute(reference, image)
    return final_transform

def register(image, reference, transform):
    return sitk.Resample(image, reference, transform, sitk.sitkLinear, 0.0, image.GetPixelID())

class Register(torch.nn.Module):
    """Registers one image to a reference image"""
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def __call__(self, image):
        registration_transform = get_registration_transform(image, self.reference)
        return register(image, self.reference, registration_transform)

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
    labels = np.array(list(sizes.keys()))
    labels_sorted = labels[largest_sizes_idx]

    lungs = ws == labels_sorted[0]
    if len(labels_sorted) > 1 and sizes[labels_sorted[1]] > sizes[labels_sorted[0]]/32:
        print("Combining largest segment with 2nd largest segment")
        lungs |= ws == labels_sorted[1]

    return lungs

class RegisterUsingLungSegmentation(torch.nn.Module):
    """"""
    def __init__(self, ref_img):
        super().__init__()
        self.ref_lungs = segment_lungs(ref_img)

    def forward(self, image):
        image_lungs = segment_lungs(image)
        registration_transform = get_registration_transform(image_lungs, self.ref_lungs)
        return register(image, self.ref_lungs, registration_transform)

class ResampleIsotropic(torch.nn.Module):
    """Resamples the image to isotropic spacing (the output size will vary between images)"""
    def __init__(self):
        super().__init__()
        self._resampler = sitk.ResampleImageFilter()
        self._resampler.SetInterpolator(sitk.sitkLinear)
        self._resampler.SetOutputSpacing((1,1,1))

    def forward(self, image):
        self._resampler.SetOutputDirection(image.GetDirection())
        self._resampler.SetOutputOrigin(image.GetOrigin())

        # calclulate sizing to fit all of resampled image
        size = image.GetSize()
        spacing = image.GetSpacing()
        size = (int(size[0] * spacing[0]),
                int(size[1] * spacing[1]),
                int(size[2] * spacing[2]))
        self._resampler.SetSize(size)

        return self._resampler.Execute(image)

class SITKImageToTensor(torch.nn.Module):
    """Converts a SITK image to a PyTorch tensor"""
    def __init__(self, add_channels_dim=True):
        super().__init__()
        self.add_channels_dim = add_channels_dim

    def forward(self, image):
        array = sitk.GetArrayFromImage(image)
        tensor = torch.tensor(array)
        if self.add_channels_dim:
            tensor.unsqueeze_(dim=0)
        return tensor

if __name__=="__main__":
    dataset = LIDCIDRIDataset(path="LIDC-IDRI")

    # add registration of the lungs to a reference image before resampling
    # to isotropic spacing and turning the image into a tensor
    ref_img = dataset._load_image("01-01-2000-CT THORAX WCONTRAST-56900")
    dataset.transform = torchvision.transforms.Compose([
            RegisterUsingLungSegmentation(ref_img),
            ResampleIsotropic(),
            SITKImageToTensor()
        ])

    img, label = next(dataset.__iter__())
    print("Label:", label)
    print("Tensor shape:", img.shape)
    print("Tensor:")
    print(img)

    import matplotlib.pyplot as plt
    plt.imshow(img[0,180,:,:], cmap='gray')
