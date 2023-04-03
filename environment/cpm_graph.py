# reference: https://github.com/Kruszylo/shifting-bottleneck
import networkx as nx


class CPM(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self._dirty = True
        self._makespan = -1
        self._criticalPath = None

    def add_node(self, *args, **kwargs):
        self._dirty = True
        super().add_node(*args, **kwargs)

    def add_nodes_from(self, *args, **kwargs):
        self._dirty = True
        super().add_nodes_from(*args, **kwargs)

    def add_edge(self, *args):  # , **kwargs):
        self._dirty = True
        super().add_edge(*args)  # , **kwargs)

    def add_edges_from(self, *args, **kwargs):
        self._dirty = True
        super().add_edges_from(*args, **kwargs)

    def remove_node(self, *args, **kwargs):
        self._dirty = True
        super().remove_node(*args, **kwargs)

    def remove_nodes_from(self, *args, **kwargs):
        self._dirty = True
        super().remove_nodes_from(*args, **kwargs)

    def remove_edge(self, *args):  # , **kwargs):
        self._dirty = True
        super().remove_edge(*args)  # , **kwargs)

    def remove_edges_from(self, *args, **kwargs):
        self._dirty = True
        super().remove_edges_from(*args, **kwargs)

    def _forward(self):
        # clear
        for j in self:
            self.nodes[j]['C'] = 0

        # update
        for n in nx.topological_sort(self):
            S = max([self.nodes[j]['C'] for j in self.predecessors(n)], default=0)
            self.nodes[n]['S'] = S
            self.nodes[n]['C'] = S + self.nodes[n]['prt']

    def _backward(self):  # update Cp, Sp to compute slack
        # clear
        for j in self:
            self.nodes[j]['Sp'] = self._makespan

        # update
        for n in reversed(list(nx.topological_sort(self))):
            Cp = min([self.nodes[j]['Sp'] for j in self.successors(n)], default=self._makespan)
            self.nodes[n]['Cp'] = Cp
            self.nodes[n]['Sp'] = Cp - self.nodes[n]['prt']

    def _compute_critical_path(self):
        G = set()
        for n in self:
            if self.nodes[n]['C'] == self.nodes[n]['Cp']:
                G.add(n)
        self._criticalPath = self.subgraph(G)

    # @property
    def makespan(self):
        if self._dirty:
            self._update()
        return self._makespan

    # @property
    def critical_path(self):
        if self._dirty:
            self._update()
        return self._criticalPath

    def _update(self):
        self._forward()
        self._makespan = max(nx.get_node_attributes(self, 'C').values())

        # self._backward()
        # self._compute_critical_path()
        self._dirty = False


