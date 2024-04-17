#!/bin/bash
species=md
outsuffix=1
seqwin=201
kfold=5

space=" "
machine_method_1="LGBM XGB RF SVM"
encode_method_1="DNC TNC RCKmer CKSNAP PseEIIP binary ENAC ANF NCP EIIP" 
w2v_encode="W2V_1_64_128_40_1"
encode_method_1w=${encode_method_1}$space${w2v_encode}

machine_method_2=""
encode_method_2="" 
encode_method_2w=${encode_method_2}$space${w2v_encode} 

cd ..
main_path=`pwd`
echo ${main_path}
cd program

cd ml
########## MACHINE LEARNING ##########
train_path=${main_path}/data/dataset/cross_val
test_csv=${main_path}/data/dataset/independent_test/independent_test.csv
result_path=${main_path}/data/result_${species}

# convnetional encodings
lag=5
weight=0.1
lamada=2
kmer=1

for machine_method in ${machine_method_1}
do
    for encode_method in ${encode_method_1}
    do
    kmer=1
    w2v_model=None
    size=-1
    epochs=-1
    window=-1
    sg=-1
    echo ${machine_method} ${encode_method}
    python ml_train_test_64.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --type RNA --seqwin ${seqwin} --fold ${kfold} --lag ${lag}  --weight ${weight}  --lamadaValue ${lamada} --kmer ${kmer} 
    done

    for kmer in 1
    do
    echo ${machine_method} kmer_${kmer}
    size=64 
    epochs=128 
    sg=1 
    window=40 
    w2v_model=${main_path}/data/w2v_model/rna_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt
    encode_method=W2V_${kmer}_${size}_${epochs}_${window}_${sg}
    python ml_train_test_64.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method} --encode ${encode_method} --type RNA --seqwin ${seqwin} --fold ${kfold} --w2vmodel ${w2v_model} --kmer ${kmer} 
    done

done
cd ..


########## ENSEMBLE LEARNING ##########
top_list=1,4,8,12,16,20,24,28,32,36,40,44
outfile=result_${outsuffix}.xlsx

echo evaluation
python analysis_622.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} 

echo fusion
python ml_fusion_52.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --top_list ${top_list}

echo output
python csv_xlsx_34.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --outfile ${outfile}





