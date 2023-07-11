# MLm5C

# Develoepment environment
 >python 3.8.8  
 >anaconda 4.11.0  
 >pandas 1.2.5  
 >numpy 1.20.3  
 >scikit-learn 1.1.2  
 >lightgbm 3.2.1  
 >xgboost  1.4.2  

# Execution
# 1 Setting directories
Users must keep the directory structure

# 2 Construction of dataset
Before simulation, users build dataset files for cross validataion and independent test: 
>sh data_prep.sh
  
# 3 Baseline model
Encoding: DNC TNC RCKmer ENAC binary CKSNAP NCP ANF EIIP PseEIIP PSTNPss  
Machine learning: RF SVM XGB LGBM 
## 3-1 Training and testing of baseline models
$cd program/network  
$sh main.sh  
(ml_train_test_m5C.py)

## 3-2 Prediction performance visualization
To see the prediction results, users must set the input parameters in "analysis.py".  
$cd program  
$python analysis.py  

# 4 Combination of selected baseline models
To make a combined model, users must perform "analysis.py" to make the input files necessary for "ml_fusion_m5C.py" and then set the input parameters in "ml_fusion_m5C.py." 
$cd program/network  
$python ml_fusion_m5C.py  
The final prediction results by the combined model is saved in /data/result/combine.csv

# References on RNA encodings
https://ilearn.erc.monash.edu/ 

# History
from py31/pred_ml_m5C directory in kurata14, base environment

