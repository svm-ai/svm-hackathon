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

    def __init__(
            self,
            aa_embed_dim,
            dna_embed_dim,
            pdb_embed_dim,
            go_embed_dim,
            msa_embed_dim,
            text_embed_dim,
            in_embed_dim,
            out_embed_dim
    ):
        super().__init__()
        self.modality_trunks = self._create_modality_trunk(
            aa_embed_dim,
            dna_embed_dim,
            pdb_embed_dim,
            go_embed_dim,
            msa_embed_dim,
            text_embed_dim,
            out_embed_dim
        )
        self.modality_heads = self._create_modality_head(
            in_embed_dim,
            out_embed_dim,
        )
        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )


    def _create_modality_trunk(
            self,
            aa_embed_dim,
            dna_embed_dim,
            pdb_embed_dim,
            go_embed_dim,
            msa_embed_dim,
            text_embed_dim,
            in_embed_dim
    ):
        """
        The current layers are just a proof of concept
        and are subject to the opinion of others.
        :param aa_embed_dim:
        :param dna_embed_dim:
        :param pdb_embed_dim:
        :param go_embed_dim:
        :param msa_embed_dim:
        :param text_embed_dim:
        :param in_embed_dim:
        :return:
        """
        modality_trunks = {}

        modality_trunks[ModalityType.AA] = nn.Sequential(
            nn.Linear(aa_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )

        modality_trunks[ModalityType.DNA] = nn.Sequential(
            nn.Linear(dna_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )

        modality_trunks[ModalityType.PDB] = nn.Sequential(
            nn.Linear(pdb_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )
        
        modality_trunks[ModalityType.GO] = nn.Sequential(
            nn.Linear(go_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )

        modality_trunks[ModalityType.MSA] = nn.Sequential(
            nn.Linear(msa_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )

        modality_trunks[ModalityType.TEXT] = nn.Sequential(
            nn.Linear(text_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, in_embed_dim),
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_head(
            self,
            in_embed_dim,
            out_embed_dim
    ):
        modality_heads = {}

        modality_heads[ModalityType.AA] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DNA] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.PDB] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.GO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.MSA] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_embed_dim, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_embed_dim, out_embed_dim, bias=False),
        )
        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}
        modality_postprocessors[ModalityType.AA] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.DNA] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.PDB] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.GO] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.MSA] = Normalize(dim=-1)


        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        """
        input = {k_1: [v],k_n: [v]}
        for key in input
            get trunk for key
            forward pass of value in trunk
            get projection head of key
            forward pass of value in projection head
            append output in output dict
        return { k_1, [o], k_n: [o]}
        """

        outputs = {}

        for modality_key, modality_value in inputs.items():


            modality_value = self.modality_trunks[modality_key](
                modality_value
            )

            modality_value = self.modality_heads[modality_key](
                modality_value
            )
             
            modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )
            outputs[modality_key] = modality_value

        return outputs


def create_proteinbind(pretrained=False):
    """
    The embedding dimensions here are dummy
    :param pretrained:
    :return:
    """
    model = ProteinBindModel(
        aa_embed_dim=480,
        dna_embed_dim=1280,
        pdb_embed_dim=128,
        go_embed_dim=600,
        msa_embed_dim=768,
        text_embed_dim=768,
        in_embed_dim=256,
        out_embed_dim=256
    )

    if pretrained:
        #get path from config
        PATH = 'best_model.pth'

        model.load_state_dict(torch.load(PATH))

    return model