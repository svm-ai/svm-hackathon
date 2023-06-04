import anc2vec
import pandas as pd
import ast
import numpy as np

embeds = anc2vec.get_embeddings()
text=pd.read_csv("/home/bazarova1/Uniprot-extracted_comments_and_GO_terms.csv")

for i in range(text.shape[0]):
    
    go_cc=ast.literal_eval(text['GO_terms_CC'][i])
    go_bp=ast.literal_eval(text['GO_terms_BP'][i])
    go_mf=ast.literal_eval(text['GO_terms_MF'][i])
    
    emb_cc=[]
    emb_bp=[]
    emb_mf=[]

    for j in range(len(go_cc)):
        if go_cc[j] in embeds.keys():
            emb_cc.append(embeds[go_cc[j]])
        else:
            emb_cc.append(np.zeros(200))

    if len(emb_cc)>0:
        emb_cc_ave=np.mean(np.stack(emb_cc),0)
    else:
        emb_cc_ave=np.zeros(200)

    for j in range(len(go_bp)):
        if go_bp[j] in embeds.keys():
            emb_bp.append(embeds[go_bp[j]])
        else:
            emb_bp.append(np.zeros(200))
    
    if len(emb_bp)>0:
        emb_bp_ave=np.mean(np.stack(emb_bp),0)
    else:
        emb_bp_ave=np.zeros(200)

    for j in range(len(go_mf)):
        if go_mf[j] in embeds.keys():
            emb_mf.append(embeds[go_mf[j]])
        else:
            emb_mf.append(np.zeros(200))

    if len(emb_mf)>0:
        emb_mf_ave=np.mean(np.stack(emb_mf),0)
    else:
        emb_mf_ave=np.zeros(200)

    emb_fin=np.concatenate((emb_cc_ave,emb_bp_ave,emb_mf_ave))

    np.save("/p/haicluster/go_embeddings/"+text['primaryAccession'][i]+".npy",emb_fin)
