# ProteinBind

ML-Driven Bioinformatics for Protein Mutation Analysis
This repository contains the source code and resources for our bioinformatics project aimed at identifying how gene/protein mutations alter function and which mutations can be pathogenic. Our approach is ML-driven and utilizes a multimodal contrastive learning framework, inspired by the ImageBind model by MetaAI.

## Project Goal

Our goal is to develop a method that can predict the effect of sequence variation on the function of genes/proteins. This information is critical for understanding gene/protein function, designing new proteins, and aiding in drug discovery. By modeling these effects, we can better select patients for clinical trials and modify existing drug-like molecules to treat previously untreated populations of the same disease with different mutations.

## Model Description

Our model uses contrastive learning across several modalities including amino acid (AA) sequences, Gene Ontology (GO) annotations, multiple sequence alignment (MSA), 3D structure, text annotations, and DNA sequences.

We utilize the following encoders for each modality:

- AA sequences: ESM v1/v2 by MetaAI
- Text annotations: Sentence-BERT (SBERT)
- 3D structure: ESMFold by MetaAI
- DNA nucleotide sequence: Nucleotide-Transformer
- MSA sequence: MSA-transformer


The NT-Xent loss function is used for contrastive learning.

## Getting Started

Clone the repository and install the necessary dependencies. Note that we will assume you have already installed Git Large File Storage (Git LFS) as some files in this repository are tracked using Git LFS.


## Contributing
Contributions are welcome! Please read the contributing guidelines before getting started.

## License

This project is licensed under the terms of the MIT license.