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

#folders=glob.glob("/p/haicluster/data_openfold/*")
my_file = open("your_file.txt", "r")
data = my_file.read()
folders = data.split("\n")
my_file.close()
L=len(folders)
#j=5
rp=np.random.permutation(range(110000,150000))
for i in rp[0:20000]:
    name=folders[i].split('/')[-1]
    name1=glob.glob(folders[i]+'/pdb/*')
    if len(name1)>0 and name1[0].split('.')[-1]=='pdb':
        with open(folders[i]+'/pdb/'+name+'.pdb', 'r') as file:
            p=file.read()
            emb=pg.embed_coords(str2coords(p))
            torch.save(emb,'/p/haicluster/pdb_embeddings/'+name+'.t')




