import numpy as np
import os
import pickle

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
except Exception:
    plt = None
    nx = None

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute the cosine similarity between two vectors. Supports input as lists or NumPy arrays.

    Args:
        vec1 (list[float] or np.ndarray): The first vector.
        vec2 (list[float] or np.ndarray): The second vector.

    Returns:
        float: Cosine similarity, ranging from -1 to 1.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Only one-dimensional vectors are supported.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0 

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

def export_graph_png(graph_pkl_path: str, png_path: str) -> bool:
    """Load a NetworkX graph from pickle and export to a PNG.

    Args:
        graph_pkl_path: path to the pickled NetworkX graph (e.g., task_layer_graph.pkl)
        png_path: output PNG path (e.g., graph.png)

    Returns:
        bool: True iff export succeeded.
    """
    try:
        if plt is None or nx is None:
            print("export_graph_png: matplotlib or networkx not available")
            return False
        if not os.path.exists(graph_pkl_path):
            print(f"export_graph_png: graph pickle not found -> {graph_pkl_path}")
            return False
        with open(graph_pkl_path, 'rb') as f:
            G = pickle.load(f)
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        ax = fig.add_subplot(111)
        if len(G) == 0:
            ax.text(0.5, 0.5, "Empty graph", ha='center', va='center')
        else:
            pos = nx.spring_layout(G, seed=42)
            try:
                has_cluster = all('cluster_id' in G.nodes[n] for n in G.nodes)
            except Exception:
                has_cluster = False
            if has_cluster:
                clusters = [G.nodes[n].get('cluster_id', 0) for n in G.nodes]
                unique = sorted(set(clusters))
                cmap = plt.cm.get_cmap('tab20', max(1, len(unique)))
                color_map = {cid: i for i, cid in enumerate(unique)}
                node_color = [color_map[G.nodes[n].get('cluster_id', 0)] for n in G.nodes]
                nx.draw(G, pos, node_size=120, node_color=node_color, cmap=cmap,
                        edge_color='gray', width=0.6, alpha=0.9, with_labels=False, ax=ax)
            else:
                nx.draw(G, pos, node_size=120, edge_color='gray', width=0.6,
                        alpha=0.9, with_labels=False, ax=ax)
        os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"export_graph_png: failed -> {e}")
        return False

if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [1, 2, 3]
    print(cosine_similarity(vec1, vec2))