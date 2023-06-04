import progres as pg
from io import StringIO
import os
import sys
from urllib import request
import torch
import glob
import numpy as np

def str2coords(s):
    coords = []
    for line in s.split('\n'):
        if (line.startswith("ATOM  ") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL"):
            break
    return coords

file_pdb='A0A0A0B5Q0.pdb'

with open(file_pdb, 'r') as file:
    p=file.read()
    emb=pg.embed_coords(str2coords(p))
    
print(emb)
