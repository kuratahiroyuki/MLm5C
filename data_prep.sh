#!/bin/sh

main_path=$(pwd)

infile1=${main_path}/data/dataset/train_m5C_201_2.txt
infile2=${main_path}/data/dataset/test_m5C_201_2.txt

kfold=5
seqwin=201

outfile1=${main_path}/data/dataset/train_m5C_${seqwin}.txt
outfile2=${main_path}/data/dataset/test_m5C_${seqwin}.txt
cv_path=${main_path}/data/dataset
test_fasta=${main_path}/data/dataset/independent_test/independent_test.fa
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv


python formatting_split_31.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2} --seqwin ${seqwin}
python train_division_1.py --infile1 ${outfile1} --cv_path ${cv_path} --kfold ${kfold} 
python test_fasta.py --infile1 ${outfile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} 

