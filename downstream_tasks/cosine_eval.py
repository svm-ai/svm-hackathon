import pandas as pd 
import numpy as np
from tqdm import tqdm 
import os

from sentence_transformers import SentenceTransformer, util

model_name = 'output/facebook-esm2_t12_35M_UR50D-v0/'
model = SentenceTransformer(model_name)
print(f"Max Sequence Length: {model.max_seq_length}")

df_test_name = 'esnpgo-seq-cleaned-test.csv'
df_test = pd.read_csv(f'data/{df_test_name}')

mut_idx_cutoff = 1024
df_test = df_test.query(f'mut_idx < {mut_idx_cutoff}') # only eval on seqs where mutation happens within tokenized seq

# Two lists of sentences
batch_size = 4
n_batches = len(df_test) // batch_size + 1
cosine_scores = []
for df_i in tqdm(np.array_split(df_test, n_batches), total=n_batches):
    sentences1 = df_i['seq_wt'].values
    sentences2 = df_i['seq_mt'].values

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores += [util.cos_sim(emb1, emb2)[0][0].cpu().numpy()  
                        for emb1, emb2 in zip(embeddings1, embeddings2)]

df_test[f'cosine-{model_name}'] = cosine_scores
df_test.to_csv(f'data/{os.path.splitext(df_test_name)[0]}-mutidx_cut={mut_idx_cutoff}-pred.csv') 

# test_samples = []
# for i, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
#     inp_example = InputExample(texts=[row['seq_wt'], row['seq_mt']], label=float(row['label']))
#     test_samples.append(inp_example) 
    
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=2, show_progress_bar=True, name='test-sim')
# test_evaluator(model, output_path=f'{model_save_path}')

# test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, batch_size=2, show_progress_bar=True, name='test-binary')
# test_evaluator(model, output_path=f'{model_save_path}')


