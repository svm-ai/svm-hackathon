import os
import numpy as np
import pandas as pd

def get_seq_mt(s):
    aa_wt = s.Variant[0] # wildtype (reference) AA
    aa_mt = s.Variant[-1] # mutated AA

    mut_idx = s.Variant[1:-1] # mutation index
    assert mut_idx.isdigit()
    mut_idx = int(mut_idx) - 1
    if mut_idx >= len(s.Sequence):
        return pd.Series([None, None, None, None], index=['seq_mt', 'aa_wt', 'aa_mt', 'mut_idx'])
    
    if s.Sequence[mut_idx] != aa_wt:  
        return pd.Series([None, None, None, None], index=['seq_mt', 'aa_wt', 'aa_mt', 'mut_idx']) 
    
    seq_mt = s.Sequence[:mut_idx] + aa_mt + s.Sequence[mut_idx+1:]
    
    return pd.Series([seq_mt, aa_wt, aa_mt, mut_idx], index=['seq_mt', 'aa_wt', 'aa_mt', 'mut_idx'])


file_name = 'data/esnpgo-seq.csv'
df = pd.read_csv(file_name)
df = df.dropna(axis=0).reset_index()

a = df.apply(get_seq_mt, axis=1)
df = pd.concat([df, a], axis=1)
df = df.dropna(axis=0).reset_index()

df = df.drop(['level_0', 'index'], axis=1)
df = df.rename({'Sequence': 'seq_wt'}, axis=1)
df['label'] = df['Class'].map({'BLB': 1, 'PLP': 0})

df.to_csv(f'{os.path.splitext(file_name)[0]}-cleaned.csv')