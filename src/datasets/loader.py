# src/datasets/loader.py

import dgl
from dgl.data import FraudDataset, FraudAmazonDataset
import torch

def load_graph_dataset(name='amazon'):
    """
    Carica un dataset di grafi per la detection delle frodi tramite DGL.
    
    Args:
        name (str): 'amazon' per FraudAmazonDataset, 'yelp' per FraudDataset('yelp')
        
    Returns:
        g (DGLGraph): Il grafo caricato
        features (torch.Tensor): Le feature dei nodi
        labels (torch.Tensor): Le etichette (0: normale, 1: frode)
    """
    print(f"--- Caricamento dataset: {name} ---")
    
    if name == 'amazon':
        dataset = FraudAmazonDataset()
        g = dataset[0]
    elif name == 'yelp':
        # 'yelp' è una variante di FraudDataset
        dataset = FraudDataset('yelp')
        g = dataset[0]
    else:
        raise ValueError(f"Dataset '{name}' non supportato. Scegli tra 'amazon' o 'yelp'.")

    # In DGL, questi dataset sono spesso 'Heterograph' (multi-relazionali).
    # Per semplicità nel Federated Learning, spesso si convertono in omogenei.
    if len(g.etypes) > 1:
        print(f"Rilevate relazioni multiple ({g.etypes}). Conversione in grafo omogeneo...")
        # Trasformiamo le diverse relazioni in archi semplici mantenendo feature e label.
        # Nelle versioni recenti di DGL le chiavi restano 'feature' e 'label'.
        g = dgl.to_homogeneous(g, ndata=['feature', 'label'])

    features = g.ndata['feature']
    labels = g.ndata['label']
    
    print(f"Grafo caricato con {g.num_nodes()} nodi e {g.num_edges()} archi.")
    print(f"Numero di feature: {features.shape[1]}")
    
    return g, features, labels

# Se esegui il file direttamente, puoi fare un test rapido
if __name__ == "__main__":
    graph, feat, lab = load_graph_dataset('amazon')
    print("Test completato con successo.")