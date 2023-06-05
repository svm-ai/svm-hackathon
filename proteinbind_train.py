import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss
from proteinbind_new import DualEmbeddingDataset, create_proteinbind
from transformers import pipeline
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
test_num = 10
num_epochs = 1000
datasets = ['go', 'dna', 'text', 'msa', 'pdb']
file_paths = {x: (f"{x.capitalize()}_files/{x}_AA_embeddings.pt", f"{x.capitalize()}_files/{x.capitalize()}_embeddings.pt") for x in datasets}


def create_dataloader(file_paths, data_type, test_num=test_num, batch_size=batch_size):
    dataset = DualEmbeddingDataset(*file_paths, data_type)
    indices = list(range(len(dataset)))
    train_indices, test_indices = indices[:-test_num], indices[-test_num:]
    train_dataloader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size)
    test_dataloader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size)
    return train_dataloader, test_dataloader

dataloaders = {x: create_dataloader(file_paths[x], x) for x in datasets}
num_batches = min(len(dl[0]) for dl in dataloaders.values())
model = create_proteinbind().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = SelfSupervisedLoss(NTXentLoss())

def process_batch(model, batch, data_type):
    input_data = {
        data_type: batch[data_type].type(torch.float32).to(device),
        'aa': batch['aa'].type(torch.float32).to(device)
    }
    output = model(input_data)
    loss = loss_func(output["aa"], output[data_type]) + loss_func(output[data_type], output["aa"])
    return loss

def model_train_test(model, dataloaders, loss_func, optimizer, num_epochs, num_batches):
    for epoch in tqdm(range(num_epochs)):
        losses = []
        model.train()
        for data_type, (train_dataloader, _) in dataloaders.items():
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= num_batches: break
                optimizer.zero_grad()
                loss = process_batch(model, batch, data_type)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}: joint train loss {sum(losses) / num_batches}")
            model.eval()
            with torch.no_grad():
                test_losses = []
                for data_type, (_, test_dataloader) in dataloaders.items():
                    for batch_idx, batch in enumerate(test_dataloader):
                        if batch_idx >= num_batches: break
                        test_loss = process_batch(model, batch, data_type)
                        test_losses.append(test_loss.item())
            print(f"Epoch {epoch+1}: joint test loss {sum(test_losses) / num_batches}")
            model.train()

model_train_test(model, dataloaders, loss_func, optimizer, num_epochs, num_batches)
