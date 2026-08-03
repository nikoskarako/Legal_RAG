# Louvain clustering over embeddings ALREADY stored by your LlamaIndex PGVectorStore
# EXACT-style graph logic (k-NN on cosine sim) + per-cluster JSON export with {doc_id, distance}
# Reads DB and table from .env (PG_* and VECTOR_TABLE=laws_vector_table_v2)

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

# Optional logger compatible with your original dbg
try:
    from utils import dbg
except Exception:
    def dbg(*args, **kwargs):
        pass

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from sqlalchemy import create_engine, inspect, text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import networkx as nx
from community import community_louvain


# -------------------------------------------------
# Build SQLAlchemy URI from env
# -------------------------------------------------

def build_db_uri_from_env() -> str:
    db = os.getenv('PG_DATABASE')
    host = os.getenv('PG_HOST', 'localhost')
    port = os.getenv('PG_PORT', '5432')
    user = os.getenv('PG_USER')
    pwd = os.getenv('PG_PASSWORD', '')
    if not db or not user:
        raise RuntimeError('PG_DATABASE and PG_USER must be set in .env')
    auth = f"{user}:{pwd}@" if pwd else f"{user}@"
    return f"postgresql+psycopg2://{auth}{host}:{port}/{db}"


# -------------------------------------------------
# Load per-DOC embeddings from your PGVector table
# Mirrors your helper; handles either metadata_->>'ref_doc_id' or 'doc_id'
# -------------------------------------------------

def load_doc_embeddings_ids_only(db_uri: str, vector_table: str) -> pd.DataFrame:
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    available = inspector.get_table_names()
    # Auto-handle LlamaIndex PGVectorStore naming: it creates data_<table> and index_<table>
    if vector_table not in available and f"data_{vector_table}" in available:
        dbg(f"'{vector_table}' not found, switching to 'data_{vector_table}' as created by PGVectorStore")
        vector_table = f"data_{vector_table}"
    if vector_table not in available:
        raise RuntimeError(f"Vector table '{vector_table}' not found. Available tables: {available}")

    # Prefer ref_doc_id if present, else fallback to doc_id, else id
    query = f"""
        SELECT
            vec.embedding AS embedding,
            COALESCE(
                vec.metadata_->>'ref_doc_id',
                vec.metadata_->>'doc_id',
                vec.id::text
            ) AS doc_id
        FROM {vector_table} AS vec
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    dbg(f"Loaded {len(df)} rows from {vector_table}")

    # Embedding value may come as list, JSON string, or memoryview depending on driver
    def _to_np(x):
        if isinstance(x, str):
            try:
                return np.array(json.loads(x), dtype=np.float32)
            except Exception:
                pass
        try:
            return np.array(x, dtype=np.float32)
        except Exception:
            # final fallback
            return np.asarray(list(x), dtype=np.float32)

    df['embedding'] = df['embedding'].apply(_to_np)

    grouped = df.groupby('doc_id').agg({
        'embedding': lambda embs: np.mean(np.vstack(embs.values), axis=0),
    }).reset_index().rename(columns={'embedding': 'doc_embedding'})

    dbg(f"Aggregated to {len(grouped)} unique doc_ids")
    grouped['doc_embedding'] = grouped['doc_embedding'].apply(lambda v: np.asarray(v, dtype=np.float32))
    if len(grouped):
        dbg(f"Doc embeddings dtype sample: {grouped['doc_embedding'].iloc[0].dtype}")
    return grouped


# -------------------------------------------------
# Graph building + Louvain (exact style you used)
# -------------------------------------------------

def cluster_with_louvain_embeddings(df: pd.DataFrame, k: int = 10, min_sim: float = 0.0, resolution: float = 1.0, use_lcc: bool = True):
    arr = np.vstack(df['doc_embedding'].values).astype(np.float32, copy=False)
    dbg(f"Clustering on arr shape={arr.shape}, dtype={arr.dtype}, k={k}")

    sim_matrix = cosine_similarity(arr)

    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    for i in range(len(df)):
        neighbors = np.argsort(sim_matrix[i])[::-1][1:k+1]  # top-k (skip self)
        for j in neighbors:
            w = float(sim_matrix[i, j])
            if w < min_sim:
                continue
            if G.has_edge(i, j):
                if w > G[i][j]['weight']:
                    G[i][j]['weight'] = w
            else:
                G.add_edge(i, j, weight=w)

    if use_lcc and G.number_of_edges() > 0:
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            lcc = max(comps, key=len)
            G = G.subgraph(lcc).copy()
            dbg(f"Using LCC with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
    df['cluster'] = df.index.map(lambda idx: partition.get(idx, -1))

    counts = df['cluster'].value_counts().sort_index().to_dict()
    dbg(f"Cluster label counts: {counts}")

    return df, arr, partition


# -------------------------------------------------
# Order members by cosine distance to cluster centroid
# -------------------------------------------------

def rank_members_by_centroid(df: pd.DataFrame, arr: np.ndarray) -> Dict[int, list]:
    Xn = normalize(arr)
    ordered: Dict[int, list] = {}
    for label, group in df.groupby('cluster'):
        if label == -1:
            continue
        idxs = group.index.to_list()
        cent = Xn[idxs].mean(axis=0)
        cent /= (np.linalg.norm(cent) + 1e-12)
        items = []
        for i in idxs:
            sim = float(np.dot(Xn[i], cent))
            dist = 1.0 - sim
            items.append({'doc_id': df.loc[i, 'doc_id'], 'distance': dist})
        items.sort(key=lambda x: x['distance'])
        ordered[int(label)] = items
    return ordered


# -------------------------------------------------
# Persist clusters: one JSON per cluster in ./Clusters
# -------------------------------------------------

def save_clusters_json(clusters: Dict[int, list], output_dir: str = paths.CLUSTERS_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for cid, items in clusters.items():
        path = os.path.join(output_dir, f"cluster_{cid}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    dbg(f"Wrote {len(clusters)} cluster files to {output_dir}")


# -------------------------------------------------
# Main (no CLI; reads everything from .env)
# -------------------------------------------------

def main():
    vector_table = os.getenv('VECTOR_TABLE', 'data_laws_vector_table_v2')
    k = int(os.getenv('LOUVAIN_K', '10'))
    min_sim = float(os.getenv('LOUVAIN_MIN_SIM', '0.0'))
    resolution = float(os.getenv('LOUVAIN_RESOLUTION', '1.0'))
    use_lcc = os.getenv('LOUVAIN_USE_LCC', 'true').lower() not in {'0','false','no'}
    output_dir = os.getenv('CLUSTERS_DIR', paths.CLUSTERS_DIR)

    db_uri = build_db_uri_from_env()
    dbg(f"Connecting to {db_uri}")

    df = load_doc_embeddings_ids_only(db_uri, vector_table)
    df, arr, _ = cluster_with_louvain_embeddings(df, k=k, min_sim=min_sim, resolution=resolution, use_lcc=use_lcc)
    clusters = rank_members_by_centroid(df, arr)
    save_clusters_json(clusters, output_dir)


if __name__ == '__main__':
    main()
