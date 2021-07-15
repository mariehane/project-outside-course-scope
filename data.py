from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision
import torchio

class LIDCIDRIDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, download=False):
        super(LIDCIDRIDataset).__init__()

        path = Path(path)

        self.metadata = pd.read_csv(path / "LIDC-IDRI_MetaData.csv")
        self.nodules 
        self.reader = sitk.ImageSeriesReader()
    
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

# TODO: LUNA16 Dataset
# TODO: NLTC Dataset?