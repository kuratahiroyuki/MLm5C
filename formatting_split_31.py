#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 15, 2024
@author: Kurata Laboratory
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile1', type=str, help='file')
    parser.add_argument('--infile2', type=str, help='file')
    parser.add_argument('--outfile1', type=str, help='file')
    parser.add_argument('--outfile2', type=str, help='file')
    parser.add_argument('--seqwin', type=int, help='file')
    
    infile1 = parser.parse_args().infile1
    infile2 = parser.parse_args().infile2
    outfile1 = parser.parse_args().outfile1
    outfile2 = parser.parse_args().outfile2
    seqwin = parser.parse_args().seqwin
    
    df1 = pd.read_csv(infile1, delimiter=',', header=None, names=['seq', 'label'], index_col = None)
    df2 = pd.read_csv(infile2, delimiter=',', header=None, names=['seq', 'label'], index_col = None)

    half = int((seqwin-1)/2)
    center = int((201-1)/2)
    
    XS_train = df1['seq'].tolist()    
    X_train = [ XS[center-half:center+half+1] for XS in XS_train]
    y_train = df1['label'].tolist()

    XS_test = df2['seq'].tolist()
    X_test = [ XS[center-half:center+half+1] for XS in XS_test]
    y_test = df2['label'].tolist()
    
    df_train=pd.DataFrame([], columns=["seq", "label"])
    df_test =pd.DataFrame([], columns=["seq", "label"])
    df_train['seq'] = X_train
    df_train['label'] = y_train
    df_test['seq'] = X_test
    df_test['label'] = y_test
    print(df_test)
    
    df_train.to_csv(outfile1, header=None, index=None)
    df_test.to_csv(outfile2, header=None, index=None)
    

