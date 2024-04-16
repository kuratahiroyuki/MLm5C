environment mldl (CERVO)
environment base (kurata14)

from m5C_2_201 (CERVO)
seqwin=201

Dataset
sequence 201 used from genRNA
train_m5C_201_2.txt 15428
test_m5C_201_2.txt   3858

Automatic program of data preparation, training, testing, and stacking (fusion)
Data preparation
data_prep.sh
 formatting_split_31.py

Training and testing
/program/ml
main_21.sh
 ml_train_test_64.py
 
/program/network
main_1.sh
 dl_train_test_1.py 

The whole process
/program
process.sh


