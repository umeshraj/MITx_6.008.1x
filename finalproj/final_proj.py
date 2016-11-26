import copy
import numpy as np
import collections

# DO NOT MODIFY THIS FUNCTION
def convert_tree_as_set_to_adjacencies(tree):
    """
    This snippet of code converts between two representations we use for
    edges (namely, with Chow-Liu it suffices to just store edges as a set of
    pairs (i, j) with i < j), whereas when we deal with learning tree
    parameters and code Sum-Product it will be convenient to have an
    "adjacency list" representation, where we can query and find out the list
    of neighbors for any node. We store this "adjacency list" as a Python
    dictionary.

    Input
    -----
    - tree: a Python set of edges (where (i, j) being in the set means that we
        don't have to have (j, i) also stored in this set)

    Output
    ------
    - edges: a Python dictionary where `edges[i]` gives you a list of neighbors
        of node `i`
    """
    edges = {}
    for i, j in tree:
        if i not in edges:
            edges[i] = [j]
        else:
            edges[i].append(j)
        if j not in edges:
            edges[j] = [i]
        else:
            edges[j].append(i)
    return edges


# DO NOT MODIFY THIS CLASS
class UnionFind():
    def __init__(self, nodes):
        """
        Union-Find data structure initialization sets each node to be its own
        parent (so that each node is in its own set/connected component), and
        to also have rank 0.

        Input
        -----
        - nodes: list of nodes
        """
        self.parents = {}
        self.ranks = {}

        for node in nodes:
            self.parents[node] = node
            self.ranks[node] = 0

    def find(self, node):
        """
        Finds which set/connected component that a node belongs to by returning
        the root node within that set.

        Technical remark: The code here implements path compression.

        Input
        -----
        - node: the node that we want to figure out which set/connected
            component it belongs to

        Output
        ------
        the root node for the set/connected component that `node` is in
        """
        if self.parents[node] != node:
            # path compression
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, node1, node2):
        """
        Merges the connected components of two nodes.

        Inputs
        ------
        - node1: first node
        - node2: second node
        """
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:  # only merge if the connected components differ
            if self.ranks[root1] > self.ranks[root2]:
                self.parents[root2] = root1
            else:
                self.parents[root1] = root2
                if self.ranks[root1] == self.ranks[root2]:
                    self.ranks[root2] += 1


def compute_empirical_distribution(values):
    """
    Given a sequence of values, compute the empirical distribution.

    Input
    -----
    - values: list (or 1D NumPy array or some other iterable) of values

    Output
    ------
    - distribution: a Python dictionary representing the empirical distribution
    """
    distribution = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    N = len(values)
    for tmp in values:
        distribution[tmp] = distribution.get(tmp, 0) + 1/N

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return distribution


def compute_empirical_mutual_info_nats(var1_values, var2_values):
    """
    Compute the empirical mutual information for two random variables given a
    pair of observed sequences of those two random variables.

    Inputs
    ------
    - var1_values: observed sequence of values for the first random variable
    - var2_values: observed sequence of values for the second random variable
        where it is assumed that the i-th entries of `var1_values` and
        `var2_values` co-occur

    Output
    ------
    The empirical mutual information *in nats* (not bits)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    empirical_mutual_info_nats = 0.0
    var1_dist = compute_empirical_distribution(var1_values)
    var2_dist = compute_empirical_distribution(var2_values)

    var12_values = list(zip(var1_values, var2_values))
    var12_dist = compute_empirical_distribution(var12_values)

    tmp = 0
    for xy, p_xy in var12_dist.items():
        p_x = var1_dist[xy[0]]
        p_y = var2_dist[xy[1]]
        tmp = p_xy * np.log(p_xy/(p_x * p_y))
        empirical_mutual_info_nats += tmp
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return empirical_mutual_info_nats



def chow_liu(observations):
    """
    Run the Chow-Liu algorithm.

    Input
    -----
    - observations: a 2D NumPy array where the i-th row corresponds to the
        i-th training data point

        *IMPORTANT*: it is assumed that the nodes in the graphical model are
        numbered 0, 1, ..., up to the number of variables minus 1, where the
        number of variables in the graph is determined from `observations` by
        looking at `observations.shape[1]`

    Output
    ------
    - best_tree: a Python set consisting of edges that are in a Chow-Liu tree
        (note that if edge (i, j) is in this set, then edge (j, i) should not
        be in the set; also, for grading purposes, please present the edges
        so that for an edge (i, j) in this set, i < j
    """
    best_tree = set()  # we will add in edges to this set
    num_obs, num_vars = observations.shape
    union_find = UnionFind(range(num_vars))

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    mutual_info_mtx = np.zeros((num_vars, num_vars))
    for row_idx in np.arange(start=0, stop=num_vars):
        for col_idx in np.arange(start=row_idx+1, stop=num_vars):
            var1_values = observations[:, row_idx]
            var2_values = observations[:, col_idx]
            mutual_info = compute_empirical_mutual_info_nats(var1_values,
                                                             var2_values)
            mutual_info_mtx[row_idx, col_idx] = mutual_info


    num_edges = len(best_tree)
    mim = mutual_info_mtx
    while num_edges < num_vars-1:
        # find max
        max_idx = np.where(mim == mim.max())
        i = int(max_idx[0][0])
        j = int(max_idx[1][0])
        #try adding edge
        if union_find.find(i) != union_find.find(j):
            best_tree.add((i, j))
            union_find.union(i, j)
        num_edges = len(best_tree)
        #remove max from mutual info
        mim[i, j] = 0

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return best_tree


def compute_empirical_conditional_distribution(var1_values, var2_values):
    """
    Given two sequences of values (corresponding to samples from two
    random variables), compute the empirical conditional distribution of
    the first variable conditioned on the second variable.

    Inputs
    ------
    - var1_values: list (or 1D NumPy array or some other iterable) of values
        sampled from, say, $X_1$
    - var2_values: list (or 1D NumPy array or some other iterable) of values
        sampled from, say, $X_2$, where it is assumed that the i-th entries of
        `var1_values` and `var2_values` co-occur
    Output
    ------
    - conditional_distributions: a dictionary consisting of dictionaries;
        `conditional_distributions[x_2]` should be the dictionary that
        represents the conditional distribution $X_1$ given $X_2 = x_2$
    """
    conditional_distributions = {x2: {} for x2 in set(var2_values)}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    for x2 in set(var2_values):
        idx = var2_values == x2
        x1_values = var1_values[idx]
        dist = compute_empirical_distribution(x1_values)
        conditional_distributions[x2] = dist

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return conditional_distributions


def learn_tree_parameters(observations, tree, root_node=0):
    """
    Learn a collection of node and edge potentials from observations that
    corresponds to a maximum likelihood estimate.

    Please use the approach presented in the course video/notes. Remember that
    the only node potential that isn't all 1's is the one corresponding to the
    root node chosen, and the edge potentials are set to be empirical
    conditional probability distributions.

    Inputs
    ------
    - observations: a 2D NumPy array where the i-th row corresponds to the
        i-th training data point

        *IMPORTANT*: it is assumed that the nodes in the graphical model are
        numbered 0, 1, ..., up to the number of variables minus 1, where the
        number of variables in the graph is determined from `observations` by
        looking at `observations.shape[1]`
    - tree: a set consisting of which edges are present (if (i, j) is in the
        set, then you don't have to also include (j, i)); note that the
        nodes must be as stated above
    - root_node: an integer specifying which node to treat as the root node

    Outputs
    -------
    - node_potentials: Python dictionary where `node_potentials[i]` is
        another Python dictionary representing the node potential table for
        node `i`; this means that `node_potentials[i][x_i]` should give the
        potential value for what, in the course notes, we call $\phi_i(x_i)$
    - edge_potentials: Python dictionary where `edge_potentials[(i, j)]` is
        a dictionaries-within-a-dictionary representation for a 2D potential
        table so that `edge_potentials[(i, j)][x_i][x_j]` corresponds to
        what, in the course notes, we call $\psi_{i,j}(x_i, x_j)$

        *IMPORTANT*: For the purposes of this project, please be sure to
        specify both `edge_potentials[(i, j)]` *and* `edge_potentials[(j, i)]`,
        where `edge_potentials[(i, j)][x_i][x_j]` should equal
        `edge_potentials[(j, i)][x_j][x_i]` -- we have provided a helper
        function `transpose_2d_table` below that, given edge potentials
        computed in one "direction" (i, j), computes the edge potential
        for the "other direction" (j, i)
    """
    nodes = set(range(observations.shape[1]))
    edges = convert_tree_as_set_to_adjacencies(tree)
    node_potentials = {}
    edge_potentials = {}

    def transpose_2d_table(dicts_within_dict_table):
        """
        Given a dictionaries-within-dictionary representation of a 2D table
        `dicts_within_dict_table`, computes a new 2D table that's also a
        dictionaries-within-dictionary representation that is the transpose of
        the original 2D table, so that:

            transposed_table[x1][x2] = dicts_within_dict_table[x2][x1]

        Input
        -----
        - dicts_within_dict_table: as described above

        Output
        ------
        - transposed_table: as described above
        """
        transposed_table = {}
        for x2 in dicts_within_dict_table:
            for x1 in dicts_within_dict_table[x2]:
                if x1 not in transposed_table:
                    transposed_table[x1] = \
                        {x2: dicts_within_dict_table[x2][x1]}
                else:
                    transposed_table[x1][x2] = \
                        dicts_within_dict_table[x2][x1]
        return transposed_table
        #edge_potentials[(neighbor, node)] = edge_potential_transposed

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    for tmp_node in nodes:
        obs_values = observations[:, root_node]
        if tmp_node == root_node:
            dist = compute_empirical_distribution(obs_values)
        else:
            dist = {x: 1 for x in set(obs_values)}
        node_potentials[tmp_node] = dist

    # compute the edge potentials
    fringe = [root_node]  # this is a list of nodes queued up to be visited next
    visited = {node: False for node in nodes}  # track which nodes are visited
    while len(fringe) > 0:
        node = fringe.pop(0)  # removes the 0th element of `fringe` and returns it
        visited[node] = True  # mark `node` as visited
        for neighbor in edges[node]:
            if not visited[neighbor]:
                # do some processing that involves the edge `(neighbor, node)` here
                par_values = observations[:, node]
                child_values = observations[:, neighbor]
                tmp_dist = compute_empirical_conditional_distribution(child_values,
                                                                      par_values)
                edge_potentials[(node, neighbor)] = tmp_dist

                # include transpose
                #edge_potential_transposed = transpose_2d_table(edge_potentials)
                #edge_potentials[(neighbor, node)] = edge_potential_transposed

                # finally after you do your processing, add `neighbor` to `fringe`
                fringe.append(neighbor)



    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return node_potentials, edge_potentials


def sum_product(nodes, edges, node_potentials, edge_potentials):
    """
    Run the Sum-Product algorithm.

    Inputs
    ------
    - nodes: Python set that consists of the nodes
    - edges: Python dictionary where `edges[i]` is a list saying which nodes
        are neighbors of node `i`
    - node_potentials: Python dictionary where `node_potentials[i]` is
        another Python dictionary representing the node potential table for
        node `i`; this means that `node_potentials[i][x_i]` should give the
        potential value for what, in the course notes, we call $\phi_i(x_i)$

        *IMPORTANT*: For the purposes of this project, the alphabets of each
        random variable should be inferred from the node potentials, so each
        node potential's dictionary's keys should tell you what the alphabet is
        (or at least the subset of the alphabet for which the probability is
        nonzero); this means that you should not use collections.defaultdict
        to produce, for instance, a dictionary with no keys that outputs 1 for
        everything here since we cannot read off what the alphabet is for the
        random variable
    - edge_potentials: Python dictionary where `edge_potentials[(i, j)]` is
        a dictionaries-within-a-dictionary representation fyesor a 2D potential
        table so that `edge_potentials[(i, j)][x_i][x_j]` corresponds to
        what, in the course notes, we call $\psi_{i,j}(x_i, x_j)$

        *IMPORTANT*: For the purposes of this project, please be sure to
        specify both `edge_potentials[(i, j)]` *and* `edge_potentials[(j, i)]`,
        where `edge_potentials[(i, j)][x_i][x_j]` should equal
        `edge_potentials[(j, i)][x_j][x_i]`

    Output
    ------
    - marginals: Python dictionary where `marginals[i]` gives the marginal
        distribution for node `i` represented as a dictionary; you do *not*
        need to store entries that are 0
    """
    marginals = {}
    messages = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #
    root_node = list(node_potentials.keys())[0]
    rt2lf_path = get_path(nodes, edges, root_node)
    lf2rt_path = rt2lf_path[-1::-1]
    all_messages = {}
    for path in lf2rt_path:
        par, child = path
        all_messages = get_message(child, par,
                                   nodes, edges, node_potentials, edge_potentials,
                                   all_messages)

    for path in rt2lf_path:
        par, child = path
        all_messages = get_message(par, child,
                                   nodes, edges, node_potentials, edge_potentials,
                                   all_messages)

    for node in nodes:
        marginals[node] = compute_marginal(node, edges, node_potentials, all_messages)

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return marginals

def compute_marginal(node, edges, node_potentials, all_messages):
    marg = node_potentials[node]
    #to_nodes = edges[node]
    for tmp in edges[node]:
        msg = all_messages[(tmp, node)]
        for key, val in msg.items():
            marg[key] *= val

    marg = normalize_marginal(marg)

    return marg

def normalize_marginal(marg):
    # normalize margina
    Z = 0
    for key, val in marg.items():
        Z += val
    for key, val in marg.items():
        marg[key] /= Z
    return marg


def get_message(from_node, to_node,
                nodes, edges, node_potentials, edge_potentials,
                all_messages):
    phi_from = node_potentials[from_node]
    psi = edge_potentials[(from_node, to_node)]
    # get prev messages
    # compute messages
    msg = {}
    phi_to = node_potentials[to_node]
    prev_msg_prod = get_prev_msg_prod(from_node, to_node, edges,
                                      node_potentials, all_messages)

    for key_to in phi_to.keys():
        tmp = 0
        for key_from in phi_from.keys():
            tmp += phi_from[key_from] * psi[key_from][key_to] * prev_msg_prod[key_from]
        msg[key_to] = tmp
    all_messages[(from_node, to_node)] = msg
    return all_messages


def get_prev_msg_prod(from_node, to_node, edges,
                      node_potentials,
                      all_messages):
    prev_edges = get_prev_edges(from_node, to_node, edges)
    prod_msg = {}

    phi_to = node_potentials[to_node]
    for key_to in phi_to.keys():
        prod_msg[key_to] = 1

    if prev_edges:
        for edge in prev_edges:
            tmp_dict = all_messages[edge]
            for key, val in tmp_dict.items():
                prod_msg[key] = prod_msg.get(key, 1) * val
    return prod_msg


def get_prev_edges(from_node, to_node, edges):
    prev_nodes = edges.get(from_node, None)
#    prev_nodes.remove(to_node)
    prev_msg = []
    for prev_node in prev_nodes:
        if prev_node != to_node:
            prev_msg.append((prev_node, from_node))
    return prev_msg

def get_path(nodes, edges, root_node):
#    nodes = {1, 2, 3}
#    edges = {1: [2], 2: [1, 3], 3: [2]}
#    nodes = {1, 2, 3, 4, 5}
#    edges = {1: [2, 3], 2: [1, 4, 5], 3: [1], 4: [2], 5: [2]}
#    nodes = {1, 2, 3, 4, 5, 6, 7}
#    edges = {1: [2, 3], 2: [1, 4, 5], 3: [1, 6, 7],
#             4: [2], 5: [2], 6:[3], 7: [3]}
#    root_node = 1
    edges_list = []

    fringe = collections.deque([root_node])
    visited = {node: False for node in nodes}  # track which nodes are visited
    while len(fringe) > 0:
        #node = fringe.pop(0)  # removes the 0th element of `fringe` and returns it
        node = fringe.popleft()
        visited[node] = True  # mark `node` as visited
        for neighbor in edges[node]:
            if not visited[neighbor]:
                # do some processing that involves the edge `(neighbor, node)` here
                edges_list.append((node, neighbor))
                # finally after you do your processing, add `neighbor` to `fringe`
                fringe.append(neighbor)
    return edges_list

def test_sum_product1():
    """
    Below is the example from
    "Exercise: The Sum-Product Algorithm - A Numerical Calculation"
    where we have conditioned on $X_1 = 0$ (conditioning can be done by setting
    the node potential to be all 0's except for at the observed value -- see
    below in the code for how this is done with `node_potentials[1]`)
    """
    nodes = {1, 2, 3}
    edges = {1: [2], 2: [1, 3], 3: [2]}

    node_potentials = {1: {0: 1, 1: 0}, 2: {0: 1, 1: 1}, 3: {0: 1, 1: 1}}
    edge_potentials = {(1, 2): {0: {0: 5, 1: 1}, 1: {0: 1, 1: 5}},
                       (2, 1): {0: {0: 5, 1: 1}, 1: {0: 1, 1: 5}},
                       (2, 3): {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
                       (3, 2): {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}}}

    marginals = sum_product(nodes, edges, node_potentials, edge_potentials)
    print('Your output:', marginals)
    print('Expected output:',
          {1: {0: 1.0},
           2: {0: 0.8333333333333334, 1: 0.16666666666666666},
           3: {0: 0.16666666666666666, 1: 0.8333333333333334}})

    node_potentials = {1: {0: 1, 1: 1}, 2: {0: 1, 1: 1}, 3: {0: 1, 1: 1}}
    print(compute_marginals_given_observations(nodes, edges,
                                               node_potentials,
                                               edge_potentials,
                                               observations={1: 0}))


def test_sum_product2():
    """
    Below is the example from
    "Homework Problem: Blue Green Tree" and
    "Homework Problem: Blue Green Tree, Continued"
    """
    nodes = {1, 2, 3, 4, 5}
    edges = {1: [2, 3], 2: [1, 4, 5], 3: [1], 4: [2], 5: [2]}

    node_potentials = {1: {'blue': 0.5, 'green': 0.5},
                       2: {'blue': 0.5, 'green': 0.5},
                       3: {'blue': 0.6, 'green': 0.4},
                       4: {'blue': 0.8, 'green': 0.2},
                       5: {'blue': 0.8, 'green': 0.2}}
    edge_potentials = {(1, 2): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (2, 1): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (1, 3): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (3, 1): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (2, 4): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (4, 2): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (2, 5): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}},
                       (5, 2): {'blue': {'blue': 0, 'green': 1},
                                'green': {'blue': 1, 'green': 0}}}

    marginals = sum_product(nodes, edges, node_potentials, edge_potentials)
    print('Your output:', marginals)
    print('Expected output:',
          {1: {'blue': 0.9142857142857144, 'green': 0.08571428571428572},
           2: {'blue': 0.08571428571428569, 'green': 0.9142857142857143},
           3: {'blue': 0.08571428571428572, 'green': 0.9142857142857144},
           4: {'blue': 0.9142857142857143, 'green': 0.0857142857142857},
           5: {'blue': 0.9142857142857143, 'green': 0.0857142857142857}})


def compute_marginals_given_observations(nodes, edges, node_potentials,
                                         edge_potentials, observations):
    """
    For a given choice of nodes, edges, node potentials, and edge potentials,
    and also observed values for specific nodes, we can compute marginals
    given the observations. This can actually be done by just modifying the
    node potentials and then calling the Sum-Product algorithm.

    Inputs
    ------
    - nodes, edges, node_potentials, edge_potentials: see documentation for
        sum_product()
    - observations: a dictionary where each key is a node and the value for
        the key is what the observed value for that node is (for example,
        `{1: 0}` means that node 1 was observed to have value 0)

    Output
    ------
    marginals, given the observations (see documentation for the output of
    sum_product())
    """
    new_node_potentials = {}

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    #

    #
    # END OF YOUR CODE
    # -------------------------------------------------------------------------

    return sum_product(nodes,
                       edges,
                       new_node_potentials,
                       edge_potentials)


def main():
    # get coconut oil data
    observations = []
    with open('coconut.csv', 'r') as f:
        for line in f.readlines():
            pieces = line.split(',')
            if len(pieces) == 5:
                observations.append([int(pieces[1]),
                                     int(pieces[2]),
                                     int(pieces[3]),
                                     int(pieces[4])])
    observations = np.array(observations)

    best_tree = chow_liu(observations)
    print(best_tree)

    node_potentials, edge_potentials = learn_tree_parameters(observations,
                                                             best_tree)
    print(node_potentials)
    print(edge_potentials)

#    marginals = compute_marginals_given_observations(
#        {0, 1, 2, 3},
#        convert_tree_as_set_to_adjacencies(best_tree),
#        node_potentials,
#        edge_potentials,
#        observations={1: +1, 2: +1})
#    print(marginals)
#    print()

    print('[Sum-Product tests based on earlier course material]')
    test_sum_product1()
    test_sum_product2()


if __name__ == '__main__':
    main()
#    # testing
#    get_path(None, None, None)

#    from sklearn.metrics import mutual_info_score
#    var1_values = np.random.randint(low=-1, high=2, size=10)
#    var2_values = np.random.randint(low=-1, high=2, size=10)
#    ur = compute_empirical_mutual_info_nats(var1_values, var2_values)
#    sk = mutual_info_score(var1_values, var2_values)
#    print("Test Mutual info: UR: {0:4.4f}, SK: {1:4.4f}".format(ur, sk))