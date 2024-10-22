import numpy as np 
import networkx as nx

class DirectedGraphGenerator:
    """
    -------------------------------------------------------------------
    Create the structure of a Directed (potentially cyclic) graph
    -------------------------------------------------------------------
    Args:
    nodes (int)            : Number of nodes in the graph.
    expected_density (int) : Expected number of edges per node. 
    """

    def __init__ (self, nodes=30, expected_density=3, enforce_dag=False):
        self.nodes = nodes
        self.expected_density = expected_density  
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
        self.p_node = expected_density/nodes
        self.cyclic = None
        self.enforce_dag = enforce_dag

    def __call__(self, permute=False):
        vertices = np.arange(self.nodes)
        for i in range(self.nodes):
            if self.enforce_dag:
                possible_parents = vertices[:i]
            else:
                possible_parents = np.setdiff1d(vertices, i)
            num_parents = np.random.binomial(n=len(possible_parents), p=self.p_node)
            parents = np.random.choice(possible_parents, size=num_parents, replace=False)

            # In networkx, the adjacency matrix is such that
            # the rows denote the parents and the columns denote the children. 
            # That is, W_ij = 1 ==> i -> j exists in the graph.
            self.adjacency_matrix[parents, i] = 1
            if self.enforce_dag:
                if permute:
                    ind = np.random.permutation(vertices)
                    col_perm_matrix = np.zeros((self.nodes, self.nodes))
                    row_perm_matrix = np.zeros((self.nodes, self.nodes))
                    for i in range(self.nodes):
                        col_perm_matrix[np.where(ind == i)[0][0], i] = 1
                        row_perm_matrix[i, np.where(ind == i)[0][0]] = 1
                    
                    self.adjacency_matrix = row_perm_matrix @ self.adjacency_matrix @ col_perm_matrix
                    

            self.g = nx.DiGraph(self.adjacency_matrix)
            self.cyclic = not nx.is_directed_acyclic_graph(self.g)

        return self.g
