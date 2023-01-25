import networkx as nx
import numpy as np
import qubogen


def build_qubo(d, task=1, prob_con=0.5, seed=42):
    graph = nx.fast_gnp_random_graph(n=d, p=prob_con, seed=seed)
    edges = np.array(list([list(e) for e in graph.edges]))
    n_nodes = len(np.unique(np.array(edges).flatten()))
    g = qubogen.Graph(edges=edges, n_nodes=n_nodes)

    if task == 1:
        Q = qubogen.qubo_max_cut(g)
    elif task == 2:
        Q = qubogen.qubo_mvc(g)
    elif task == 3:
        Q = qubogen.qubo_set_pack(g)

    def func(I):
        return ((I @ Q) * I).sum(axis=1)

    return func
