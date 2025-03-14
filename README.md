# TLforCE

This project aims to explore the question: *can transfer learning be used to better predict the Coulombic efficiency (CE) of LMBs based on the electrolyte properties?*

This repository contains the data and code necessary to reproduce results discussed in our paper.

#### All data used is contained in the data folder. 
* CE_X.csv and CE_y.csv contains the features/input and target CE values of the CE dataset from [Kim et al. PNAS (2023)](https://www.pnas.org/doi/10.1073/pnas.2214357120). This is directly obtained from the supplementary materials and unprocessed, except to divide into two separate files.

* conductivity_X.csv and conductivity_y.csv contains the features/input and target conductivity values of the electrolyte conductivity dataset from [de Blasio et al. Scientific Data (2024)](https://www.nature.com/articles/s41597-024-03575-8)
    * The code necessary to obtain these processed features are contained in cond_data_proc.py, and the original unprocessed dataset is contained in CALiSol-23 Dataset.csv
    * The elements.csv file contains relevant molecular information used in processing the conductivity data set

The code to construct and evaluate the linear regression, random forest and nueral network models are contained in linreg_main.py, rforest_main.py and nn_main.py. 

The three files can be directly run to output the results discussed in the paper. The required python packages are listed in requirements.txt. Parameters such as the random seed, neural network architecture, training hyperparameters, etc. can be adjusted directly in the code if desired.


