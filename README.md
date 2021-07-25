# Crowd counting without density map!
The input image with an arbitrary shape is divided into cells of $64 \times 64$. 
A batch of cells enter to the network which tries to find the head locations in each cell.
Each cell is considered in a grid of $16 \times 16$. In each small grid cell of $4 \times 4$, one head location will be detected.
The loss is a classification loss between a binary map ground-truth and predicted head locations.
Therefore, it can localize the head location as well. 
The final count is the summation of predicted heads in all cells.


## Usage
Run train.py

## Results 

| Datasets        | MAE   | Method              | Parameters                        | STAT MAE |
|:--------------- |:-----:|:--------------------|:----------------------------------|:--------:|
| Shanghai A      | 102   | Grid-CSRNet         | LR=1e-4, cell=64, epoch=50        |    61    |
| Shanghai B      | 19    | Grid-CSRNet-FC      | LR=1e-4, cell=64, epoch=20        |     8    | 
| Shanghai A      | 97    | Grid-CSRNet-FC      | LR=1e-4, cell=64, epoch=40        |    61    |  
| Shanghai A      | 91    | Grid-CSRNet-2FC     | LR=1e-6, cell=128, epoch=70       |    61    | 
| Shanghai B      | 10.2  | Grid-CSRNet-2FC-2ch | LR=1e-7, cell=128, epoch=250      |     8    |    

<details>
  <summary>Table Notes (click to expand)</summary>
  
  * Grid-CSRNet denotes grid-based CSRNet architecture
  * FC denotes fully connected layer 
  * 2ch: output includes 2 channels, one 1/8 input and for localization, another one carries the 'count' for each cell
  * LR = learning rate
</details>
