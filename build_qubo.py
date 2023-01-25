import networkx as nx
import numpy as np
import qubogen


def build_qubo(d, task=1, prob_con=0.5, seed=42):
    graph = nx.fast_gnp_random_graph(n=d, p=prob_con, seed=seed)
    edges = np.array(list([list(e) for e in graph.edges]))
    n_nodes = len(np.unique(np.array(edges).flatten()))
    g = qubogen.Graph(edges=edges, n_nodes=n_nodes)

    if task == 1:     # Quadratic Knapsack Problem
        v = np.diag(np.random.random(d)) / 3.
        a = np.random.random(d)
        b = np.mean(a)
        Q = qubogen.qubo_qkp(v, a, b)
        print(Q)
    elif task == 2:   # Max-Cut Problem
        Q = qubogen.qubo_max_cut(g)
    elif task == 3:   # Minimum Vertex Cover Problem (MVC)
        Q = qubogen.qubo_mvc(g)
    def func(I):
        return ((I @ Q) * I).sum(axis=1)

    return func
