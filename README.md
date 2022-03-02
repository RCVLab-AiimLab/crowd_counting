# [Multiscale Crowd Counting and Localization by Multitask Point Supervision](https://arxiv.org/abs/2202.09942)
## Abstract:
We propose a multitask approach for crowd counting and person localization in a unified framework. As the detection and localization tasks are well-correlated and can be jointly tackled, our model benefits from a multitask solution by learning multiscale representations of encoded crowd images, and subsequently fusing them. In contrast to the relatively more popular density-based methods, our model uses point supervision to allow for crowd locations to be accurately identified. We test our model on two popular crowd counting datasets, ShanghaiTech A and B, and demonstrate that our method achieves strong results on both counting and localization tasks, with MSE measures of 110.7 and 15.0 for crowd counting and AP measures of 0.71 and 0.75 for localization, on ShanghaiTech A and B respectively. Our detailed ablation experiments show the impact of our multiscale approach as well as the effectiveness of the fusion module embedded in our network. 

## Results:
We evaluated our method against previous crowd counting methods using MSE, MAE and AP measures. The results from the proposed method are presented below:


|              |   ShanghaiTech A  |   ShanghaiTech A  |
|              |  Counting    Loc. |  Counting    Loc. |
|     Method   | MAE    MSE    AP  | MAE    MSE    AP  |
| -------------| ----------------- | ----------------- |
| Cross-scene  |181.1  277.7   -   |32.0   49.8    -   |
| MCNN         |110.2  173.2   -   |26.4   41.3    -   |
| LC-ResFCN    |  -      -     -   |25.89    -     -   |
| LC-PSPNet    |  -      -     -   |21.61    -     -   |
| RAZNet       |75.2   133.0  0.69 |13.5   25.4   0.69 |
| RAZNet+      |71.6   120.1  0.69 |9.9    15.6   0.71 |
| DD-CNN       |71.9   111.2  0.65 |  -      -     -   |
| Deep-Stacked |94.0   150.6   -   |18.7   31.9    -   |
| CSRNet       |68.2   115.0   -   |10.6   16      -   |
| Ours         |71.4   110.7  0.71 |9.6    15.0   0.75 | 


### Visualization:

Visualized samples of detection and localization. Yel-
low and red points denote detected and ground truth head lo-
cations, respectively:

![](imgs/vis.png)


## Dependencies
[PyTorch](https://pytorch.org)

This code is tested under Ubuntu 18.04, CUDA 11.2, with one NVIDIA Titan RTX GPU.
Python 3.8.8 version is used for development.


## Datasets
[ShanghaiTech](https://www.kaggle.com/tthien/shanghaitech)

To preprocess ShanghaiTech dataset:
```
python make_dateset.py
```

## To train the code:
```
python train.py
```
We also provide trained models:

[Trained models](https://queensuca-my.sharepoint.com/:f:/g/personal/hd53_queensu_ca/Ercs-ffjKR5Jj7-AhnzXfQEB10Es-Yiyl5tSkc2bM_6XPw?e=T5LgaK)



## to test the code:
```
python eval.py
```

## Citation
Please cite our paper if you use code from this repository:
```
@article{zand2022Multiscale,
  title={Multiscale Crowd Counting and Localization By Multitask Point Supervision},
  author={Zand, Mohsen, Damirchi, Haleh, Farley, Andrew, Molahasani, Mahdiyar, Greenspan, Michael, and Etemad, Ali},
  journal={IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2022.},
  year={2022}
}
```


## References
[CSRNet](https://github.com/leeyeehoo/CSRNet)
