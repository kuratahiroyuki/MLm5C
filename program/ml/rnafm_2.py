
import torch
import pandas as pd
import numpy as np
import fm as fm

rna_dict = {'A':4,'C':5,'G':6,'U':7,'-':19}


def pad_input_fas(filename, seqwin):
    sequence = []
    rna_fm_token = []
    with open(filename,'r') as f:
        lines = f.readlines() 
    for i in range(len(lines)):
        if i%2 == 1:
            seq = lines[i]

            if len(seq) > seqwin:
                seq = seq[0:seqwin]
                seq = seq.ljust(seqwin, '-')
            sequence.append(seq)
            rna_fm_token.append([rna_dict[res] for res in seq])
    df = pd.DataFrame({'seq':sequence,'token':rna_fm_token} )

    return df


def rna_fm_encoding(rna_fm_tokens): 
    rna_fm_tokens = torch.tensor(rna_fm_tokens)
    if rna_fm_tokens.ndim == 1:
        rna_fm_tokens = rna_fm_tokens.unsqueeze(0)

    #rna_fm, _ =  fm.pretrained.rna_fm_t12()
    rna_fm, _ =  fm.pretrained.rna_fm_t12(model_location="/home/kurata/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth") 
    #rna_fm, _ = fm.pretrained.rna_fm_t12(model_location="/home/kurata/myproject/py31/pred_ml_m5C/program/network/fm/RNA-FM_pretrained.pth")
    
    rna_fm.eval()
    with torch.no_grad():
        results = rna_fm(rna_fm_tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
    for param in rna_fm.parameters():
        param.detach_()
        #print(param)
    encodings = results["representations"][12].numpy()

    B,L,F = encodings.shape
    print(f'B, L, K = {B} {L} {F}') #B, L, K = 64 41 640
    encodings = encodings.reshape(B, -1)
    print(encodings.shape)

    return encodings


if __name__=='__main__':

    infile="example_rna_1.fasta"
    seqwin=20

    df = pad_input_fas(infile, seqwin)  #df seq, token, label
    rna_fm_tokens = torch.tensor(df['token'].values.tolist())

    encodings = rna_fm_encoding(rna_fm_tokens)

    encodings = encodings.numpy()

    print('encodings {}'.format(encodings)) 
    print('encodings {}'.format(encodings.shape)) #[B,L,640]

