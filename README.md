# Crowd Counting


__Requirements__
- Python 3.8
- PyTorch 1.8
- CUDA 10.2


__Dataset__

- [Download](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI)
- change the path in line 19 in make_dataset.py to the Shanghai dataset path you have downloaded
- Run 'make_dataset.py'

__More info about dataset generation__

Grund truths are in *.mat and *.h5 files
there are two fields in *.mat: 
- 1) location: it shows the location of each person (or car)
- 2) number: it's the number of people (or cars)
- in make_dataset.py, an image-sized matrix 'k' with all zeros values is defined. All 'locations' are assigned '1' in 'k'. 
- density of matrix 'k' is obtained using a 'gaussian_filter'.  
- obtained density is saved in a 'h5' file in the 'density' field.  



__Train__

- change the path in lines 57 and 58 in train.py to the Shanghai dataset path you have downloaded
- Run 'train.py'