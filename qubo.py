from neal import SimulatedAnnealingSampler
import networkx as nx
import numpy as np
import qubogen


def qubo_build_function(Q):
    def func(I):
        return ((I @ Q) * I).sum(axis=1)
    return func


def qubo_build_matrix(d, prob_con=0.3, seed=42):
    graph = nx.fast_gnp_random_graph(n=d, p=prob_con, seed=seed)
    edges = np.array(list([list(e) for e in graph.edges]))
    n_nodes = len(np.unique(np.array(edges).flatten()))
    g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
    Q = qubogen.qubo_mvc(g) # qubo_max_cut(g)
    return Q


def qubo_solve_baseline(Q):
    # TODO: add restriction on requests to target functions
    response = SimulatedAnnealingSampler().sample_qubo(Q)
    assert len(np.unique(response.data_vectors['energy'])) == 1
    i_min = list(response.first[0].values())
    y_min = response.data_vectors['energy'][0]
    return y_min
