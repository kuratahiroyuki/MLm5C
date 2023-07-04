#!/bin/sh

program_path=$(pwd)
data_path=${program_path}/data/dataset
infile1=${data_path}/train-5mC-mm.txt
infile2=${data_path}/ind-5mC-mm.txt

outfile1=${data_path}/train_m5C.txt
outfile2=${data_path}/test_m5C.txt

test_fasta=${data_path}/independent_test/independent_test.fa
test_csv=${data_path}/independent_test/independent_test.csv
kfold=5


python ${program_path}/formatting.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2}

python ${program_path}/train_division.py --infile1 ${outfile1} --datapath ${data_path} --kfold ${kfold} 

python ${program_path}/test_fasta.py --infile1 ${outfile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} 



