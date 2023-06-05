from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ModalityType = SimpleNamespace(
    AA="aa",
    DNA="dna",
    PDB="pdb",
    GO="go",
    MSA="msa",
    TEXT="text",
)

class Config:
    EMBED_DIMS = {
        ModalityType.AA: 480,
        ModalityType.DNA: 1280,
        ModalityType.PDB: 128,
        ModalityType.GO: 600,
        ModalityType.MSA: 768,
        ModalityType.TEXT: 768,
    }
    IN_EMBED_DIM = 256
    OUT_EMBED_DIM = 256

def create_layers(embed_dim, in_embed_dim):
    return nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(512, in_embed_dim),
    )

def create_head(in_embed_dim, out_embed_dim):
    return nn.Sequential(
        nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
        nn.Dropout(p=0.1),
        nn.Linear(in_embed_dim, out_embed_dim, bias=False),
    )

class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

class EmbeddingDataset(Dataset):
    """
    The main class for turning any modality to a torch Dataset that can be passed to
    a torch dataloader. Any modality that doesn't fit into the __getitem__
    method can subclass this and modify the __getitem__ method.
    """
    def __init__(self, sequence_file_path, embeddings_file_path, modality):
        self.sequence = pd.read_csv(sequence_file_path)
        self.embedding = torch.load(embeddings_file_path)
        self.modality = modality

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        sequence = self.sequence.iloc[idx, 0]
        embedding = self.embedding[idx]
        return {"aa": sequence, self.modality: embedding}

class DualEmbeddingDataset(Dataset):
    """
    The main class for turning any modality to a torch Dataset that can be passed to
    a torch dataloader. Any modality that doesn't fit into the __getitem__
    method can subclass this and modify the __getitem__ method.
    """
    def __init__(self, sequence_embeddings_file_path, embeddings_file_path, modality):
        self.sequence_embedding = torch.load(sequence_embeddings_file_path)
        self.embedding = torch.load(embeddings_file_path)
        self.modality = modality

    def __len__(self):
        return len(self.sequence_embedding)

    def __getitem__(self, idx):
        sequence_embedding = self.sequence_embedding[idx]
        embedding = self.embedding[idx]
        return {"aa": sequence_embedding, self.modality: embedding}

class ProteinBindModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_trunks = self._create_modality_trunk()
        self.modality_heads = self._create_modality_head()
        self.modality_postprocessors = self._create_modality_postprocessors()

    def _create_modality_trunk(self):
        modality_trunks = {
            modality: create_layers(embed_dim, Config.IN_EMBED_DIM)
            for modality, embed_dim in Config.EMBED_DIMS.items()
        }
        return nn.ModuleDict(modality_trunks)

    def _create_modality_head(self):
        modality_heads = {
            modality: create_head(Config.IN_EMBED_DIM, Config.OUT_EMBED_DIM)
            for modality in Config.EMBED_DIMS
        }
        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self):
        modality_postprocessors = {
            modality: Normalize(dim=-1)
            for modality in Config.EMBED_DIMS
        }
        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            modality_value = self.modality_trunks[modality_key](modality_value)
            modality_value = self.modality_heads[modality_key](modality_value)
            modality_value = self.modality_postprocessors[modality_key](modality_value)
            outputs[modality_key] = modality_value
        return outputs


def create_proteinbind(pretrained=False):
    model = ProteinBindModel()
    if pretrained:
        #get path from config
        PATH = 'best_model.pth'
        model.load_state_dict(torch.load(PATH))
    return model