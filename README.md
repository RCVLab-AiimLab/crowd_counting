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
1) location: it shows the location of each person (or car)
2) number: it's the number of people (or cars)

- in make_dataset.py, an image-sized matrix 'k' with all zeros values is defined. All 'locations' are assigned '1' in 'k'. 
- density of matrix 'k' is obtained using a 'gaussian_filter'.  
- There are two types of 'gaussian_filters':
1) line 80, which is applied on part_A. It does not work with 'people_count' less than 4. 
2) line 109, which is applied on part_B. It is a standard implementation from scipy package. 
- obtained density is saved in a 'h5' file in the 'density' field. 
- You must run 'make_dataset.py' to generate the 'h5' files.




__Train__

- change the path in lines 57 and 58 in train.py to the Shanghai dataset path you have downloaded
- Run 'train.py'