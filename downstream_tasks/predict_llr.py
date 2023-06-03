import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

models = { # see https://github.com/facebookresearch/esm#main-models-you-should-use- 
    'ESM-2-15B': 'esm2_t48_15B_UR50D',
    'ESM-2-3B': 'esm2_t36_3B_UR50D',
    'ESM-1v': 'esm1v_t33_650M_UR90S_1', # take the first one from the ensemble of 5 models `esm1v_t33_650M_UR90S_{i}`
    'ESM-2-35M': 'esm2_t12_35M_UR50D'
}

model_location = 'ESM-2-35M'
model, alphabet = esm.pretrained.load_model_and_alphabet(models[model_location])
model = model.eval().cuda()

# index of the last model layer where embeddings will be extracted
if model_location == 'ESM-2-15B':
    embed_layer_idx = 48
elif model_location == 'ESM-2-3B':
    embed_layer_idx = 36
elif model_location == 'ESM-1v':
    embed_layer_idx = 33
elif model_location == 'ESM-2-35M':
    embed_layer_idx = 12

df_path = 'data/esnpgo-seq-cleaned-test-mutidx_cut=1024-pred.csv'
df = pd.read_csv(df_path)

# sample n_seq for tests equally for positive/negative labels
n_seq = 100
data_df = pd.concat([df.query('label == 0')[:n_seq//2], df.query('label == 1')[:n_seq//2]], axis=0)
data = [ (f'protein{i}', seq) for i, seq in enumerate(data_df['seq_wt']) ] 

batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# create a boolean mask based on mutation aa index 
mask = torch.zeros_like(batch_tokens, dtype=torch.bool).cuda()
mask[range(mask.shape[0]), data_df['mut_idx'].values + 1] = True # shift index by +1 for <BOS>

# mask the elements with the `mask` token
batch_tokens[mask] = alphabet.mask_idx

# single token loop (so that to fit in memory) 
token_probs = []
for i, t in enumerate(tqdm(batch_tokens)):
    # get probabilities 
    with torch.no_grad():
        t_probs = torch.log_softmax(
            model(t[None, :].cuda())["logits"], dim=-1)

    # select probabilities for masked tokens
    t_probs = torch.masked_select(t_probs[0], mask[i, :, None])
    token_probs.append(t_probs)
token_probs = torch.stack(token_probs, dim=0)

wt_aa_indices = torch.from_numpy(data_df['aa_wt'].map(alphabet.tok_to_idx).values)
mt_aa_indices = torch.from_numpy(data_df['aa_mt'].map(alphabet.tok_to_idx).values)

# wt and mt masks
wt_mask = torch.zeros_like(token_probs, dtype=torch.bool).cuda()
mt_mask = torch.zeros_like(token_probs, dtype=torch.bool).cuda()
wt_mask[range(wt_mask.shape[0]), wt_aa_indices] = True
mt_mask[range(wt_mask.shape[0]), mt_aa_indices] = True

proba_wt = torch.masked_select(token_probs, wt_mask)
proba_mt = torch.masked_select(token_probs, mt_mask)
data_df['llr'] = (proba_mt - proba_wt).cpu()

plt.hist(data_df.query('label == 0')['llr'], bins=20, range=(-15,5), histtype='step', label='P')
plt.hist(data_df.query('label == 1')['llr'], bins=20, range=(-15,5), histtype='step', label='B')
plt.legend()
plt.title(model_location)
plt.xlabel('LLR')
plt.show()
plt.savefig('llr.png')

print(roc_auc_score(data_df['label'], data_df['llr']))