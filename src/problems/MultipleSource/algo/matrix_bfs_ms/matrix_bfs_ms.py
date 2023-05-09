from typing import Iterable

from pygraphblas.types import BOOL
from pygraphblas.matrix import Matrix
from pygraphblas import descriptor
from pygraphblas import Accum, binaryop

from src.graph.graph import Graph

from src.problems.MultipleSource.MultipleSource import MultipleSourceProblem
from src.problems.MultipleSource.algo.matrix_bfs_ms.reg_automaton import RegAutomaton
from src.problems.utils import ResultAlgo



def init_masks_matrix(graph:Graph, grammar: RegAutomaton) -> Matrix:
    mask_matrix = Matrix.identity(BOOL, grammar.matrices_size, value=True)
    mask_matrix.resize(grammar.matrices_size, graph.matrices_size + grammar.matrices_size)
    return mask_matrix

def init_diag_matrices(graph:Graph, grammar: RegAutomaton):
    """
    Create a block diagonal matrices from graph and regex matrices for each symbol
    """
    diag_matrices = dict()
    for symbol in grammar.matrices:
        if symbol in graph:
            diag_matrix = Matrix.sparse(BOOL, graph.matrices_size + grammar.matrices_size, graph.matrices_size + grammar.matrices_size)
            diag_matrix.assign_matrix(
                grammar.matrices[symbol],
                slice(0, grammar.matrices_size - 1),
                slice(0, grammar.matrices_size - 1),
            )
            diag_matrix.assign_matrix(
                graph[symbol],
                slice(grammar.matrices_size, graph.matrices_size + grammar.matrices_size - 1),
                slice(grammar.matrices_size, graph.matrices_size + grammar.matrices_size - 1),
            )

            diag_matrices[symbol] = diag_matrix

    return diag_matrices

def init_pairs_matrix(graph: Graph, grammar: RegAutomaton, len_sources: int) -> Matrix:
    pairs_matrix = Matrix.sparse(BOOL, len_sources * grammar.matrices_size, graph.matrices_size + grammar.matrices_size)

    for i in range(len_sources):
        pairs_matrix.assign_matrix(
            init_masks_matrix(graph, grammar),
            rindex=slice(i * grammar.matrices_size, i * grammar.matrices_size + grammar.matrices_size - 1),
        )
    
    return pairs_matrix

class MatrixBFSMSAlgo(MultipleSourceProblem):

    def prepare(self, graph: Graph, grammar: RegAutomaton):
        self.graph = graph
        self.graph.load_bool_graph()
        self.grammar = grammar
        self.diag_matrices = init_diag_matrices(graph, grammar)

    def prepare_for_solve(self):
        pass

    def solve(self, sources: Iterable):
        # initialize matrices for multiple source bfs
        ident = init_masks_matrix(self.graph, self.grammar)
        vect = ident.dup()
        found = ident.dup()
        
        # fill start states
        for reg_start_state in self.grammar.start_states:
            for gr_start_state in sources:
                found[reg_start_state, self.grammar.matrices_size + gr_start_state] = True

        # matrix which contains newly found nodes on each iteration
        found_on_iter = found.dup()

        acc = found.dup()

        # Algo's body
        not_empty = True
        iter = 0
        while not_empty:
            iter += 1
            # for each symbol we are going to store if any new nodes were found during traversal.
            # if none are found, then 'not_empty' flag turns False, which means that no matrices change anymore
            # and we can stop the traversal
            not_empty_for_at_least_one_symbol = False
            
            # print(f"\n--- Level: {iter} ---")
            # print(f"vect  (size={vect.shape}):\n{vect}\n")

            vect.assign_matrix(found_on_iter, mask=vect, desc=descriptor.RC)
            vect.assign_scalar(True, mask=ident)

            # print(f"found_on_iter  (size={found_on_iter.shape}):\n{found_on_iter}\n")

            # stores found nodes for each symbol
            found_on_iter.assign_matrix(ident)

            for symbol in self.grammar.matrices:
                if symbol in self.graph:
                    with BOOL.ANY_PAIR:
                        found = vect.mxm(self.diag_matrices[symbol])

                    old_acc = acc.dup()
                    # print(f"Found[{symbol}] (size={found.shape}):\n{found}\n")

                    with Accum(binaryop.MAX_BOOL):
                        # extract left (grammar) part of the masks matrix and rearrange rows
                        i_x, i_y, _ = found.extract_matrix(col_index=slice(0, self.grammar.matrices_size - 1)).to_lists()
                        for i in range(len(i_y)):
                            found_on_iter.assign_row(i_y[i], found.extract_row(i_x[i]))
                            acc.assign_row(i_y[i], found.extract_row(i_x[i]))
                    
                    # check if new nodes were found. if positive, switch the flag
                    if not old_acc.iseq(acc):
                        not_empty_for_at_least_one_symbol = True

                    # print(f"Acc[{symbol}]  (size={acc[symbol].shape}):\n{acc[symbol]}\n")

            not_empty = not_empty_for_at_least_one_symbol

        # get nvals from acc matrix
        nvals = 0
        for i in range(self.grammar.matrices_size):
            if i not in self.grammar.final_states and i not in self.grammar.start_states:
                nvals += acc.extract_row(i).nvals - 1

        i_x, i_y, _ = acc.to_lists()
        # print(f"ix: {i_x}, total = {len(i_x)}")
        parsed = [j - self.grammar.matrices_size for j in i_y if j > self.grammar.matrices_size]
        #print(f"iy: {parsed}, total = {len(i_y)}. parsed = {len(parsed)}")
        return ResultAlgo(acc, iter), len(parsed)

class MatrixBFSMSPairsAlgo(MultipleSourceProblem):

    def prepare(self, graph: Graph, grammar: RegAutomaton):
        self.graph = graph
        #self.graph.load_bool_graph()
        self.grammar = grammar
        self.diag_matrices = init_diag_matrices(graph, grammar)

    def prepare_for_solve(self):
        pass

    def solve(self, sources: Iterable):
        # initialize matrices for multiple source bfs
        ident = init_pairs_matrix(self.graph, self.grammar, len(sources))
        vect = ident.dup()
        found = ident.dup()
        
        # fill start states
        for reg_start_state in self.grammar.start_states:
            for i, gr_start_state in enumerate(sources):
                found[
                    i * self.grammar.matrices_size + reg_start_state,
                    self.grammar.matrices_size + gr_start_state
                ] = True

        # matrix which contains newly found nodes on each iteration
        found_on_iter = found.dup()

        acc = found.dup()

        # Algo's body
        not_empty = True
        iter = 0
        while not_empty and iter < self.graph.matrices_size * self.grammar.matrices_size:
            iter += 1
            # for each symbol we are going to store if any new nodes were found during traversal.
            # if none are found, then 'not_empty' flag turns False, which means that no matrices change anymore
            # and we can stop the traversal
            not_empty_for_at_least_one_symbol = False
            
            # print(f"\n--- Level: {iter} ---")
            # print(f"vect  (size={vect.shape}):\n{vect}\n")

            vect.assign_matrix(found_on_iter, mask=vect, desc=descriptor.RC)
            vect.assign_scalar(True, mask=ident)

            # print(f"found_on_iter  (size={found_on_iter.shape}):\n{found_on_iter}\n")

            # stores found nodes for each symbol
            found_on_iter.assign_matrix(ident)

            for symbol in self.grammar.matrices:
                if symbol in self.graph:
                    with BOOL.ANY_PAIR:
                        found = vect.mxm(self.diag_matrices[symbol])

                    old_acc = acc.dup()
                    # print(f"Found[{symbol}] (size={found.shape}):\n{found}\n")
                    
                    with Accum(binaryop.MAX_BOOL):
                        # extract left (grammar) part of the masks matrix and rearrange rows
                        for s in range(len(sources)):
                            shift = s * self.grammar.matrices_size
                            i_x, i_y, _ = found.extract_matrix(row_index=slice(shift, shift + self.grammar.matrices_size -1), 
                                                               col_index=slice(0, self.grammar.matrices_size - 1)).to_lists()
                            for i in range(len(i_y)):
                                found_on_iter.assign_row(shift + i_y[i], found.extract_row(shift + i_x[i]))
                                acc.assign_row(shift + i_y[i], found.extract_row(shift + i_x[i]))
                    
                    # check if new nodes were found. if positive, switch the flag
                    if not old_acc.iseq(acc):
                        not_empty_for_at_least_one_symbol = True

                    # print(f"Acc[{symbol}]  (size={acc[symbol].shape}):\n{acc[symbol]}\n")

            not_empty = not_empty_for_at_least_one_symbol

        # get nvals from acc matrix
        pairs = []
        nvals = 0
        for s in range(len(sources)):
            for i in range(self.grammar.matrices_size):
                if i not in self.grammar.final_states and i not in self.grammar.start_states:
                    nvals += acc.extract_row(s * self.grammar.matrices_size + i).nvals - 1
                    for j in (acc.extract_row(s * self.grammar.matrices_size + i,
                    ).I):
                        if j >= self.grammar.matrices_size:
                            pairs.append((sources[s], j - self.grammar.matrices_size))

        # i_x, i_y, _ = acc.to_lists()
        # print(f"ix: {i_x}, total = {len(i_x)}")
        # print(f"iy: {i_y}, total = {len(i_y)}")
        return ResultAlgo(acc, iter), nvals, pairs
