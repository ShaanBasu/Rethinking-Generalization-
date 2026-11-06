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

**COmplications**: To save the subset_indices.npy i first ran it in kaggle so it made a copy of a  file with the indices and then i downloaded and saved it locally then pushed it to git. 