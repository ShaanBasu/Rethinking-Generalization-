# Research Log

## 2026-11-06 - Phase 1: Kaggle Environment check
**Goal:** Confirm Kaggle can pull my repo and see a GPU.

**Environment:** Kaggle Notebook, 2x Tesla T4 (15 GB each), driver 580.178.04.
Using GPU 0 only. Python 3.13.16, torch 2.11.0 (CUDA cu130), torchvision 0.26, numpy 2.1.3.

**Code version:** repo at commit 2bed1ab (first commit, only research_log.md).

**Result:** Clone worked; GPU visible. Repo is public, Kaggle internet on.

**Next:** Phase 2, data inspection and subset.

## 2026-11-07 - Phase 2: Data Handling
**Goal** Use and handle the CIFAR-10 dataset and alsohandle corruption of data labels

**Indices and Seed**: Seed used for randomly picking the indices is 0 (for now), Subset seed = 0, and also the indices which are chosed using this seed is saved in assets/subset_indices.npy. The per class count for the indices is 1000 meaning there are 1000 indices for each class. The rule for our subset of 10,000 indices is that it stratified, 1000 per class, without replacement and is the same for all 15 runs.

**Complications**: To save the subset_indices.npy i first ran it in kaggle so it made a copy of a  file with the indices and then i downloaded and saved it locally then pushed it to git. 

## 2026-11-09 - Phase 3: Inspecting Data
**Goal:** Inspect the data visually and check for mislabelled images, scrambled axes, or wrong value range.

**Images and labels:** Out of the 25 images checked in the subset created The images and their corresponding labels were all found to correct matched. (Although on a personal note the images may be a bit too pixelated for the naked eye)

**Class Count:** The class count for the subset was confirmed to be 1000 for all classes(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck),

**Shape, dtype and channel means:** For the subset the shape produced was (10000, 32, 32, 3) shape = (Batchsize, height, width, channels), dtype: uint8, The integer range for the subset_images was recorded as 0-255, and the mean for the 3 colour values for all images was approx [0.491, 0.483, 0.446]

## 2026-11-09 - Phase 4: Pre-processing the data
**Goal:** To turn raw CIFAR images of numpy array shape into the form Pytorch CNN needs: float tensors and channels first shape with each colour channel standardised and also conversion of the labels into integer type.

**Shape:** The shape of both the training and testing sets were converted from (10000,32,32,3) to (10000, 3, 32, 32), highlighting the channels first shape and their datatype was also changed from numpyarray to torch.float32. Moreover, the labels were also changed into an integer64 datatype.

**Stats:** The stats which were used to normalise the data mean: tensor([0.4913, 0.4827, 0.4464])  std: tensor([0.2469, 0.2437, 0.2624]), train mean: tensor([ 1.8436e-06, -3.3547e-06, -1.9006e-06])  std: tensor([1.0000, 1.0000, 1.0000])
test  mean: tensor([0.0120, 0.0098, 0.0151])  std: tensor([0.9989, 0.9967, 0.9970])

**Round Trip:** A round trip test was also carried out to check if the normalisation which was done to the images can be reversed into the raw form. The test was a success. We also checked if the labels matched with the subset even after changing their datatype and the labels also matched.
