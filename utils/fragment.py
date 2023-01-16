from rdkit import Chem
from copy import deepcopy
import re
from collections import deque
PLACE_HOLDER_ATOM = 80 # Hg




def canonical_frag_smi(frag_smi):
    frag_smi = re.sub(r'\[\d+\*\]', '[*]', frag_smi)
    canonical_frag_smi = Chem.CanonSmiles(frag_smi)
    return canonical_frag_smi


def wash_frag(frag):
    # frag = deepcopy(frag)
    for atom in frag.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    return frag

def reorder_frag(frag):
    frag2 = wash_frag(deepcopy(frag))
    Chem.MolToSmiles(frag2,isomericSmiles=False)
    frag_order = frag2.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder']
    return Chem.RenumberAtoms(frag,frag_order)

def get_clean_smi(mol):
    rmol =deepcopy(mol)
    for at in rmol.GetAtoms():
        at.SetAtomMapNum(0)
        at.SetIsotope(0)
    return Chem.MolToSmiles(rmol,isomericSmiles=False)

def swap_atom_order(a):
    return [a[1],a[0],a[3],a[2]]


def edge_bfs(edge_index,edge_type,source= None,dirty_type = 1):
    nodes = [0 if not source else source]
    nbrs = {}
    dirty_edge = []

    for e,t in zip(edge_index,edge_type):
        if e[0] not in nbrs:
            nbrs[e[0]] = []
        if e[1] not in nbrs:
            nbrs[e[1]] = []
        if t== dirty_type:
            dirty_edge.append(e)
        else:
            nbrs[e[0]].append(e[1])
            nbrs[e[1]].append(e[0])
    def edges_from(node):
        return [(node,nbr) for nbr in nbrs[node]]

    visited_nodes = {n for n in nodes}
    visited_edges = set()
    queue = deque([(n, edges_from(n)) for n in nodes])
    output = []
    while queue or len(output)<len(edge_index):
        if not queue:
            for e in dirty_edge:
                if e not in visited_edges:
                    if (e[0] in visited_nodes)^(e[1] in visited_nodes):
                        edge = (e[1],e[0]) if e[1] in visited_nodes else e
                    # if e[0] in visited_nodes and e[1] not in visited_nodes:
                        child = edge[1]
                        visited_nodes.add(child)
                        # print('extra',e,e[1])
                        queue.append((child, edges_from(child)))
                        visited_edges.add((edge[0],edge[1]))
                        visited_edges.add((edge[1],edge[0]))
                        output.append(edge)
                        break
                    elif e[1] in visited_nodes and e[0] in visited_nodes:
                        # print('extra',e)
                        visited_edges.add((e[0],e[1]))
                        visited_edges.add((e[1],e[0]))
                        output.append(e)
            if not queue and len(output)<len(edge_index):
                print("EDGE_BFS ERROR")
                print(edge_index,edge_type)
                break
            elif len(output)==len(edge_index):
                return output
        parent, children_edges = queue.popleft()
        for edge in children_edges:
            child = edge[1]
            if child not in visited_nodes:
                visited_nodes.add(child)
                queue.append((child, edges_from(child)))
            if (edge[0],edge[1]) not in visited_edges and (edge[1],edge[0]) not in visited_edges:
                visited_edges.add((edge[0],edge[1]))
                visited_edges.add((edge[1],edge[0]))
                output.append(edge[:2])
    # print("edge_bfs",output)
    return output