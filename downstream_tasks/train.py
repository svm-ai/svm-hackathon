from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from torch import nn
from torch.utils.data import DataLoader

import math
import pandas as pd
from tqdm import tqdm
import mlflow
from sklearn.model_selection import train_test_split

# mlflow.transformers.autolog(log_models=True) 

# model_name = 'facebook/esm1v_t33_650M_UR90S_1'
model_name = 'facebook/esm2_t12_35M_UR50D'
# model_name = 'facebook/esm2_t30_150M_UR50D'
# model_name = 'facebook/esm2_t33_650M_UR50D'
# model_name = 'facebook/esm2_t36_3B_UR50D'

model_version = 'v20'
model_save_path = 'output/'+model_name.replace("/", "-") + '-' + model_version
word_embedding_model = models.Transformer(model_name)

# head block
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

# model = pretrained + head
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])


# load train and test data
df_train = pd.read_csv('data/esnpgo-seq-cleaned-train.csv')
train_samples = []
for i, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    inp_example = InputExample(texts=[row['seq_wt'], row['seq_mt']], label=float(row['label']))
    train_samples.append(inp_example)

df_test = pd.read_csv('data/esnpgo-seq-cleaned-test.csv')
test_samples = []
for i, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    inp_example = InputExample(texts=[row['seq_wt'], row['seq_mt']], label=float(row['label']))
    test_samples.append(inp_example) 
    
# dataloader
train_batch_size = 1
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# loss and evaluator
train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-dev')

# train
num_epochs = 3
model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            # steps_per_epoch=26000,
            evaluation_steps=5000,
            # use_amp=True,
            checkpoint_path=f'{model_save_path}/checkpoints',
            checkpoint_save_steps=20000,
            output_path=model_save_path)