import numpy as np


try:
    import networkx as nx
    with_networkx = True
except Exception as e:
    with_networkx = False


try:
    import qubogen
    with_qubogen = True
except Exception as e:
    with_qubogen = False


from .bm_opt_tens import BmOptTens


class BmOptTensMaxCut(BmOptTens):
    def __init__(self, d=50, name='max_cut'):
        super().__init__(d, 2, name)
        self.desc = 'QUBO Max-Cut Problem'
        self.prep()

    def prep(self, prob_con=0.5, seed=42):
        if not with_networkx:
            self.err = 'Need "networkx" module'
            return

        if not with_qubogen:
            self.err = 'Need "qubogen" module'
            return

        graph = nx.fast_gnp_random_graph(n=self.d, p=prob_con, seed=seed)
        edges = np.array(list([list(e) for e in graph.edges]))
        n_nodes = len(np.unique(np.array(edges).flatten()))
        g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
        self.Q = qubogen.qubo_max_cut(g)

    def _f_batch(self, I):
        return ((I @ self.Q) * I).sum(axis=1)
