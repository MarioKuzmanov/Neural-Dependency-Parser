import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import graphviz
import torch

import scorer
from conllu import read_conllu


class DepGraph:

    def __init__(self, sent, add_edges=True):
        self.nodes = [tok.copy() for tok in sent]
        n = len(self.nodes)
        self.M = np.zeros(shape=(n, n))
        self.deprels = [None] * n
        self.heads = [None] * n
        if add_edges:
            for i in range(1, n):
                self.add_edge(sent[i].head,
                              i, 1.0, sent[i].deprel)

    def add_edge(self, parent, child, weight=0.0, label="_"):
        self.M[parent, child] = weight
        self.deprels[child] = label
        self.nodes[child].head = parent
        self.nodes[child].deprel = label

    def remove_edge(self, parent, child, remove_deprel=True):
        self.M[parent, child] = 0.0
        if remove_deprel:
            self.deprels[child] = None

    def edge_list(self):
        """Iterate over all edges with non-zero weights.
        """
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] != 0.0:
                    yield (self.nodes[i], self.nodes[j],
                           self.M[i, j], self.deprels[j])

    def get_children(self, node):
        for i in range(self.M.shape[1]):
            if self.M[node, i] != 0.0:
                yield i, self.M[node, i], self.deprels[i]

    def get_parents(self, node):
        for i in range(self.M.shape[0]):
            if self.M[i, node] != 0.0:
                yield i, self.M[i, node], self.deprels[node]

    def _find_cycle(self, start=0):
        """Find a cycle from the start node using an interative DFS.
        """
        stack = [start]
        visited = {start: None}
        while stack:
            node = stack.pop()
            for child, _, _ in self.get_children(node):
                if child not in visited:
                    visited[child] = node
                    stack.append(child)
                else:
                    curr, path = node, [node]
                    while curr != start:
                        curr = visited[curr]
                        path.append(curr)
                    if child in path:
                        return list(reversed(path)), visited
                    visited[child] = node
                    stack.append(child)
        return [], visited

    def find_cycle(self):
        """Find and return a cycle if exists."""
        checked = set()
        for node in range(len(self.nodes)):
            if node in checked: continue
            cycle, visited = self._find_cycle(node)
            checked.update(set(visited))
            if cycle:
                return cycle
        return []

    def todot(self):
        """Return a GraphViz Digraph - can be useful for debugging."""
        dg = graphviz.Digraph()  # graph_attr={'rankdir': 'LR'})
        for head, dep, weight, deprel in self.edge_list():
            dg.edge(head.form, dep.form, label=f"{deprel}({weight:0.2f})")
        return dg


def mst_parse(sent, id2deprel, deprel2id, score_fn):
    n = len(sent)
    mst = DepGraph(sent, add_edges=False)

    adj_matrix, labels = score_fn(sent)

    # STEP 1: Find best head and deprel foreach child
    for child in range(1, n):
        # to avoid unnecessary self-loop
        besthead = torch.argmax(adj_matrix[child]).item()
        bestrel = torch.argmax(labels[child]).item()
        maxscore = adj_matrix[child][besthead] + labels[child][bestrel]

        bestrel = id2deprel[bestrel]
        mst.add_edge(besthead, child, maxscore, bestrel)

    # STEP 2: Check whether the resulting graph contains a cycle
    cycle = mst.find_cycle()

    # STEP 3: Break cycles until they exist in the graph
    removed = set()

    while len(cycle):
        minloss, bestu, bestv, oldp, bestw, bestrel = float('inf'), None, None, None, None, ""
        for v in cycle:
            # Root node doesn't have parents
            if v != 0:
                parent, _, _ = list(mst.get_parents(v))[0]
                deprel = mst.deprels[v]

                weight = adj_matrix[parent][v] + labels[v][deprel2id[deprel]]
                for u in range(n):
                    if u == v or u in cycle or (u, v) in removed:
                        continue
                    uw = adj_matrix[u][v] + labels[v][deprel2id[deprel]]
                    if weight - uw < minloss:
                        minloss = weight - uw
                        oldp = parent
                        bestu, bestv, bestw, bestrel = u, v, uw, deprel
        # Add the cyclic edge, for which we found min_loss replacement , to the removed set
        removed.add((oldp, bestv))

        mst.remove_edge(oldp, bestv)
        mst.add_edge(bestu, bestv, bestw, bestrel)
        cycle = list(mst.find_cycle())

    # STEP 4: Resulting graph is the MST
    return mst


def evaluate(gold_sent, pred_sent):
    assert len(gold_sent) == len(pred_sent)
    n = len(gold_sent) - 1
    uas, las = 0, 0

    for i, gold in enumerate(gold_sent[1:], start=1):
        pred = pred_sent[i]
        if gold.head == str(pred.head):
            uas += 1
            if gold.deprel == pred.deprel:
                las += 1
    return uas / n, las / n


def evaluate_model(test_file):
    # train model
    neural_scorer = scorer.NeuralScorer()
    neural_scorer.load_model()

    # evaluate scorer
    n = 0
    uas_scorer, las_scorer = 0, 0
    for sent in read_conllu(test_file):
        mst_scorer = mst_parse(sent, neural_scorer.id2deprel, neural_scorer.deprel2id, score_fn=neural_scorer.scorer)
        uas_sent, las_sent = evaluate(sent, mst_scorer.nodes)
        uas_scorer += uas_sent
        las_scorer += las_sent

        n += 1

    # show results
    print(f"UAS: {round(uas_scorer / n, 3)} LAS: {round(las_scorer / n, 3)}")


if __name__ == "__main__":
    evaluate_model(test_file="UD_Bulgarian-BTB/bg_btb-ud-test.conllu")
