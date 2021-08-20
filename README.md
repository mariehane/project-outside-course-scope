# POC: Lung CT Diagnosis via. Deep Learning
Code for my Project Outside Course-scope (POC) on Deep Learning for Patients with 

## Setup
The code has been developed for Python 3.6.9 and CUDA version 11.2. It has been tested on Linux, but should work on any platform that can install the libraries.

### Data Preparation
Download all the [LIDC-IDRI Data from TCIA](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), including the CT scans (which must be downloaded using the NBIA Data Retriever) into a single folder. After this, you should have a folder structure like the following:

```
LIDC-IDRI
├── LIDC-IDRI_MetaData.csv
├── LIDC-XML-only.zip
├── lidc-idri nodule counts (6-23-2015).xlsx
├── list3.2.xls
├── tcia-diagnosis-data-2012-04-20.xls
├── manifest-1600709154662
│   ├── metadata.csv
│   ├── LIDC-IDRI
│   │   ├── LIDC-IDRI-0001
│   │   │   ├── 01-01-2000-30178
│   │   │   │   └── 3000566.000000-03192
│   │   │   │       ├── 069.xml
│   │   │   │       ├── 1-001.dcm
│   │   │   │       ├── 1-002.dcm
...
```

### Installing Dependencies
To install the required libraries, run the following command:
```
pip install -r requirements.txt
```

## Running the code
The results in the report were generated with the following three commands:
```
python main.py --model LungCNN --lidc-idri-dir LIDC-IDRI --gpus 1 --batch-size 5 --data-split-seed 42
```
```
python main.py --model LungCNN2 --lidc-idri-dir LIDC-IDRI --gpus 1 --batch-size 5 --data-split-seed 42
```
```
python main.py --model SlowR50 --lidc-idri-dir LIDC-IDRI --gpus 5 --accelerator ddp --batch-size 1 --data-split-seed 42
```
Note that the last command requires access to five GPUs with around 48GB memory each.
