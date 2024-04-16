# MLm5C

# Development environment
 >python 3.8.8   
 >anaconda 4.11.0  
 >pandas 1.2.5  
 >numpy 1.20.3  
 >lightgbm: 3.1.1  
 >xgboost: 1.7.6  
 >sklearn: 1.1.2  
 >gensim: 4.0.1  

# Execution
# 1 Setting directories
Users must keep the structure of the directories

# 2 Construction of dataset
Before simulation, users build dataset files for cross validataion and independent test:  
$sh data_prep.sh
  
# 3 Baselin model construction and ensemble model construction
$cd program  
$sh process.sh  

## 3-1 Training and testing of the baseline models
ml_train_test_64.py

## 3-2 Evaluation of the baseline models
analysis_622.py

## 3-3 Ensemble model construction
ml_fusion_52.py

## 3-4 output of result
csv_xlsx_34.py

# References on RNA encodings
https://ilearn.erc.monash.edu/

# History
from py31/pred_ml_m5C directory in kurata14, base environment

