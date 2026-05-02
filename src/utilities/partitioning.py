from __future__ import annotations

import time
import argparse
import numpy as np
import torch
import dgl
import networkx as nx


# ---------------------------------------------------------------------------
# Dirichlet partition
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels: torch.Tensor,
    g: dgl.DGLGraph,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict[int, dgl.DGLGraph]:
    """
    Partition graph nodes across clients using a Dirichlet distribution over labels.

    Each class's nodes are distributed according to Dir(alpha) proportions,
    producing non-IID splits: small alpha -> strong skew, large alpha -> near-IID.

    Args:
        labels:    Node label tensor of shape (N,).
        g:         Homogeneous DGLGraph with N nodes.
        n_clients: Number of federated clients / partitions.
        alpha:     Dirichlet concentration parameter (default 0.5).
        seed:      Random seed for reproducibility.

    Returns:
        Dictionary {client_id: DGLGraph} built with node_subgraph (no halo nodes).
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    n_classes = int(labels.max().item()) + 1
    node_part = torch.full((g.num_nodes(),), -1, dtype=torch.long)

    for c in range(n_classes):
        class_idx = torch.where(labels == c)[0].numpy()
        if len(class_idx) == 0:
            continue
        proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()
        rng.shuffle(class_idx)
        start = 0
        for client_id, count in enumerate(splits):
            node_part[class_idx[start : start + count]] = client_id
            start += count

    unassigned = (node_part == -1).nonzero(as_tuple=True)[0]
    if len(unassigned) > 0:
        node_part[unassigned] = torch.randint(0, n_clients, (len(unassigned),), generator=gen)

    return _build_client_subgraphs(g, node_part, n_clients)
# ---------------------------------------------------------------------------
# Random partition
# ---------------------------------------------------------------------------

def random_partition(
    g: dgl.DGLGraph,
    n_clients: int,
    seed: int = 42,
) -> dict[int, dgl.DGLGraph]:
    """
    Partition graph nodes uniformly at random across clients.

    Args:
        g:          Homogeneous DGLGraph.
        n_clients:  Number of partitions.
        seed:       Random seed.

    Returns:
        Dictionary {client_id: DGLGraph} built with node_subgraph (no halo nodes).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    node_part = torch.randint(0, n_clients, (g.num_nodes(),), generator=gen)
    return _build_client_subgraphs(g, node_part, n_clients)


# ---------------------------------------------------------------------------
# METIS partition (DGL)
# ---------------------------------------------------------------------------

def metis_partition(
    g: dgl.DGLGraph,
    n_clients: int,
) -> dict[int, dgl.DGLGraph]:
    """
    Partition graph with DGL METIS and rebuild no-halo client subgraphs.

    Args:
        g:          Homogeneous DGLGraph.
        n_clients:  Number of partitions.

    Returns:
        Dictionary {client_id: DGLGraph} built with node_subgraph (no halo nodes).
    """
    metis_parts = dgl.metis_partition(g, n_clients)
    node_part = torch.full((g.num_nodes(),), -1, dtype=torch.long)

    for cid in range(n_clients):
        subg = metis_parts[cid]
        gids = subg.ndata["_ID"].long()
        if "inner_node" in subg.ndata:
            gids = gids[subg.ndata["inner_node"].bool()]
        node_part[gids] = cid

    unassigned = (node_part == -1).nonzero(as_tuple=True)[0]
    if len(unassigned):
        node_part[unassigned] = torch.randint(0, n_clients, (len(unassigned),))

    return _build_client_subgraphs(g, node_part, n_clients)


# ---------------------------------------------------------------------------
# K-Means++ clustering partition
# ---------------------------------------------------------------------------

def kmeans_partition(
    g: dgl.DGLGraph,
    n_clients: int,
    features: torch.Tensor | None = None,
    seed: int = 42,
) -> dict[int, dgl.DGLGraph]:
    """
    Partition graph nodes into n_clients subgraphs using K-Means++ clustering
    on node features (sklearn.cluster.KMeans with init='k-means++').

    If no features are provided, falls back to structural features:
    (in-degree, out-degree, degree ratio) computed from the graph topology.

    This implements the 'Clustering Partition' non-IID strategy: clients
    receive nodes that are similar in feature space, producing feature
    distribution skew across clients.

    Args:
        g:          Homogeneous DGLGraph.
        n_clients:  Number of partitions (k in K-Means).
        features:   Optional node feature tensor of shape (N, d).
                    If None, uses structural graph features (degree-based).
        seed:       Random seed for K-Means++ initialization.

    Returns:
        Dictionary {client_id: DGLGraph} built with node_subgraph (no halo nodes).
    """
    from sklearn.cluster import KMeans

    gen = torch.Generator()
    gen.manual_seed(seed)

    if features is not None:
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = np.asarray(features)
    else:
        # Fallback: structural features (in-degree, out-degree, degree ratio)
        in_deg  = g.in_degrees().float()
        out_deg = g.out_degrees().float()
        ratio   = in_deg / (out_deg + 1e-6)
        X = torch.stack([in_deg, out_deg, ratio], dim=1).detach().cpu().numpy()

    # Normalize features for stable K-Means convergence
    X_mean = X.mean(axis=0, keepdims=True)
    X_std  = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    kmeans = KMeans(
        n_clusters=n_clients,
        init="k-means++",
        n_init=10,
        random_state=seed,
    )
    labels_km = kmeans.fit_predict(X_norm)
    print(f"  K-Means++ inertia: {kmeans.inertia_:.2f} | iterations: {kmeans.n_iter_}")

    node_part = torch.tensor(labels_km, dtype=torch.long)

    unassigned = (node_part < 0).nonzero(as_tuple=True)[0]
    if len(unassigned):
        node_part[unassigned] = torch.randint(0, n_clients, (len(unassigned),), generator=gen)

    return _build_client_subgraphs(g, node_part, n_clients)


# ---------------------------------------------------------------------------
# Spectral clustering partition
# ---------------------------------------------------------------------------

def spectral_partition(
    g: dgl.DGLGraph,
    n_clients: int,
    seed: int = 42,
) -> dict[int, dgl.DGLGraph]:
    """
    Partition graph nodes into n_clients subgraphs using Spectral Clustering
    (sklearn.cluster.SpectralClustering) on the graph adjacency matrix.

    Spectral clustering computes the eigenvectors of the normalized graph
    Laplacian L = D - A, then applies K-Means++ in the spectral embedding
    space. This globally minimizes the normalized cut, making it theoretically
    optimal for partitioning well-separated graph communities.

    Note: scales as O(N * n_clients) with sparse eigen-decomposition (ARPACK).
    Feasible up to ~50k nodes; for larger graphs prefer METIS.

    Args:
        g:          Homogeneous DGLGraph.
        n_clients:  Number of partitions.
        seed:       Random seed.

    Returns:
        Dictionary {client_id: DGLGraph} built with node_subgraph (no halo nodes).
    """
    from sklearn.cluster import SpectralClustering
    import scipy.sparse as sp

    gen = torch.Generator()
    gen.manual_seed(seed)

    # Build sparse adjacency matrix from DGL graph
    src, dst = g.edges()
    n = g.num_nodes()
    data = np.ones(len(src), dtype=np.float32)
    A = sp.csr_matrix(
        (data, (src.detach().cpu().numpy(), dst.detach().cpu().numpy())), shape=(n, n)
    )
    # Symmetrize (SpectralClustering requires symmetric affinity)
    A = (A + A.T).astype(np.float32)
    A.data[:] = 1.0   # binarize after symmetrization

    sc = SpectralClustering(
        n_clusters=n_clients,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    labels_sc = sc.fit_predict(A)
    print(f"  Spectral clustering done — {n_clients} partitions")

    node_part = torch.tensor(labels_sc, dtype=torch.long)

    unassigned = (node_part < 0).nonzero(as_tuple=True)[0]
    if len(unassigned):
        node_part[unassigned] = torch.randint(0, n_clients, (len(unassigned),), generator=gen)

    return _build_client_subgraphs(g, node_part, n_clients)


def _build_client_subgraphs(
    g: dgl.DGLGraph,
    node_part: torch.Tensor,
    n_clients: int,
) -> dict[int, dgl.DGLGraph]:
    """
    Build one induced subgraph per client using node_subgraph (no halo nodes).
    """
    client_graphs = {}
    for client_id in range(n_clients):
        nodes_of_this_client = torch.where(node_part == client_id)[0]
        client_graphs[client_id] = dgl.node_subgraph(g, nodes_of_this_client)
    return client_graphs


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _cross_edge_stats(
    g: dgl.DGLGraph, partitions: dict[int, dgl.DGLGraph], n_clients: int
) -> int:
    node_part_vec = torch.full((g.num_nodes(),), -1, dtype=torch.long)
    for cid in range(n_clients):
        subg  = partitions[cid]
        gids  = subg.ndata["_ID"].long()
        node_part_vec[gids] = cid
    s, d = g.edges()
    both  = (node_part_vec[s] >= 0) & (node_part_vec[d] >= 0)
    cross = (both & (node_part_vec[s] != node_part_vec[d])).sum().item()
    return cross


def _print_report(
    g: dgl.DGLGraph,
    labels: torch.Tensor,
    partitions: dict[int, dgl.DGLGraph],
    n_clients: int,
    n_classes: int,
    method: str,
    elapsed: float,
) -> None:
    cross = _cross_edge_stats(g, partitions, n_clients)
    total = g.num_edges()

    print(f"\n[{method}] Partitioning time: {elapsed:.3f}s")
    print(f"\n{'Client':<8} {'Nodes':<14} {'Label dist':<35}")
    print("-" * 58)
    for cid in range(n_clients):
        subg   = partitions[cid]
        gids   = subg.ndata["_ID"].long()
        counts = labels[gids].bincount(minlength=n_classes).tolist()
        print(f"  {cid:<6} {subg.num_nodes():<14} {counts}")
    print("-" * 58)
    print(f"Cross-partition edges: {cross} / {total} ({100 * cross / total:.1f}%)\n")


def _build_realistic_graph(n_nodes: int, n_clients: int, seed: int) -> nx.Graph:
    """
    Build a realistic heterogeneous synthetic graph using an asymmetric SBM:
    - Block sizes drawn from a Dirichlet (not equal) -> community size imbalance
    - Each block has its own internal density -> some dense hubs, some sparse periphery
    - Inter-block density scales inversely with block distance (geographic-like)

    This avoids the perfectly symmetric SBM that makes Louvain return equal-sized
    communities, producing instead a graph that resembles real-world networks.
    """
    rng = np.random.default_rng(seed)

    # Unequal block sizes: sample from Dirichlet(0.5) -> heavy imbalance
    raw_sizes = rng.dirichlet(alpha=np.full(n_clients, 0.5)) * n_nodes
    sizes = np.maximum(raw_sizes.astype(int), 5)
    sizes[-1] += n_nodes - sizes.sum()   # fix rounding
    sizes = sizes.tolist()

    # Per-block internal density: vary between 0.03 and 0.15
    p_in_per_block = rng.uniform(0.03, 0.15, size=n_clients)

    # Inter-block density: low baseline + random noise
    k = n_clients
    p_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                v = float(p_in_per_block[i])
            else:
                v = float(rng.uniform(0.001, 0.008))
            p_matrix[i][j] = v
            p_matrix[j][i] = v

    g = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=int(seed))
    return g


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: graph partitioning methods"
    )
    parser.add_argument("--n-nodes",    type=int,   default=6000)
    parser.add_argument("--n-classes",  type=int,   default=2)
    parser.add_argument("--n-clients",  type=int,   default=15)
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Dirichlet alpha (lower = more non-IID)")
    parser.add_argument("--method",     type=str,   default="both",
                        choices=["dirichlet", "random", "metis", "kmeans", "spectral", "both"])
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Realistic asymmetric synthetic graph ----
    nx_g = _build_realistic_graph(args.n_nodes, args.n_clients, args.seed)
    src_list, dst_list = zip(*nx_g.edges())
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    g = dgl.graph(
        (torch.cat([src, dst]), torch.cat([dst, src])),
        num_nodes=args.n_nodes,
    )
    g = dgl.to_simple(g)

    raw_probs = torch.tensor([0.70, 0.30][: args.n_classes], dtype=torch.float32)
    raw_probs /= raw_probs.sum()
    labels = torch.multinomial(
        raw_probs.unsqueeze(0).expand(args.n_nodes, -1), num_samples=1
    ).squeeze(1)

    print("=" * 65)
    print(f"Asymmetric SBM graph: {g.num_nodes()} nodes | {g.num_edges()} edges")
    print(f"Global label dist   : {labels.bincount(minlength=args.n_classes).tolist()}")
    print(f"Clients: {args.n_clients}  |  alpha={args.alpha}")
    print("=" * 65)

    if args.method in ("dirichlet", "both"):
        t0 = time.time()
        parts_d = dirichlet_partition(
            labels, g, args.n_clients, alpha=args.alpha, seed=args.seed
        )
        _print_report(g, labels, parts_d, args.n_clients, args.n_classes,
                      "DIRICHLET", time.time() - t0)

    if args.method in ("random", "both"):
        t0 = time.time()
        parts_r = random_partition(
            g, args.n_clients, seed=args.seed
        )
        _print_report(g, labels, parts_r, args.n_clients, args.n_classes,
                      "RANDOM", time.time() - t0)

    if args.method in ("metis", "both"):
        t0 = time.time()
        parts_m = metis_partition(g, args.n_clients)
        _print_report(g, labels, parts_m, args.n_clients, args.n_classes,
                      "METIS", time.time() - t0)

    if args.method in ("kmeans", "both"):
        t0 = time.time()
        # In the demo we pass synthetic features (random, shape N x 8)
        # On the real dataset pass g_hetero.ndata['feature']
        torch.manual_seed(args.seed)
        demo_features = torch.randn(g.num_nodes(), 8)
        parts_k = kmeans_partition(
            g, args.n_clients, features=demo_features, seed=args.seed
        )
        _print_report(g, labels, parts_k, args.n_clients, args.n_classes,
                      "KMEANS++", time.time() - t0)

    if args.method in ("spectral", "both"):
        t0 = time.time()
        parts_s = spectral_partition(g, args.n_clients, seed=args.seed)
        _print_report(g, labels, parts_s, args.n_clients, args.n_classes,
                      "SPECTRAL", time.time() - t0)