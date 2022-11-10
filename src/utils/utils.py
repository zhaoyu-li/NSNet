import networkx as nx

from itertools import combinations


def literal2v_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    return sign, v_idx


def literal2l_idx(literal):
    assert abs(literal) > 0
    sign = literal > 0
    v_idx = abs(literal) - 1
    if sign:
        return v_idx * 2
    else:
        return v_idx * 2 + 1


def safe_log(t, eps=1e-12):
    return (t + eps).log()


def write_dimacs_to(n_vars, clauses, out_path):
    with open(out_path, 'w') as f:
        f.write('p cnf %d %d\n' % (n_vars, len(clauses)))
        for clause in clauses:
            for literal in clause:
                f.write('%d ' % literal)
            f.write('0\n')


def parse_cnf_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) < 1 or tokens[0] != 'p':
            i += 1
        else:
            break
    
    if i == len(lines):
        return 0, []
    
    header = lines[i].strip().split()
    n_vars = int(header[2])
    n_clauses = int(header[3])
    clauses = []
    # for BIRD benchmark, there are some instances' n_clauses != len(clauses) 
    # for SATLIB benchmark, there are some instances end with % 0

    for line in lines[i+1:]:
        tokens = line.strip().split()
        if len(tokens) < 2 or tokens[0] == 'c' or tokens[0] == 'p': # undefined symbol, comment, duplicate header
            continue   
        clauses.append([int(s) for s in tokens[:-1]])

    return n_vars, clauses


def remove_dumplicate_literals(clauses):
    return [list(set(clause)) for clause in clauses]


def unit_propagation(clauses):   
    fixed_vars = set()
    while True:
        need_propagate = set()
        assign = {}
        for clause in clauses:
            if len(clause) == 1:
                sign, v_idx = literal2v_idx(clause[0])
                need_propagate.add(v_idx)
                assign[v_idx] = sign
        
        if len(need_propagate) == 0:
            return clauses, fixed_vars
        
        new_clauses = []
        for clause in clauses:
            if len(clause) == 1:
                continue
                
            new_clause = []
            is_sat = False
            for literal in clause:
                sign, v_idx = literal2v_idx(literal)
                if v_idx in need_propagate:
                    if sign == assign[v_idx]:
                        is_sat = True
                        break
                else:
                    new_clause.append(literal)
            
            if new_clause and not is_sat:
                new_clauses.append(new_clause)
        
        clauses = new_clauses
        fixed_vars.update(need_propagate)


def remove_fixed_vars(n_vars, clauses, fixed_vars):
    shifts = []
    shift = 0
    
    for v_idx in range(n_vars):
        shifts.append(shift)
        if v_idx in fixed_vars:
            shift += 1
    
    new_n_vars = n_vars - len(fixed_vars)
    new_clauses = [[l - shifts[abs(l)-1] if l > 0 else l + shifts[abs(l)-1] for l in clause] for clause in clauses]

    return new_n_vars, new_clauses


def add_unused_variables(n_vars, clauses):
    used_vars = set()
    new_clauses = []
    for clause in clauses:
        used_v = set([abs(l) for l in clause])
        new_clauses.append(sorted(clause, key=abs))
        used_vars.update(used_v)
    
    if len(used_vars) == n_vars:
        return new_clauses
    
    for i in range(1, n_vars + 1):
        if i not in used_vars:
            new_clauses.append([i, -i])
        
    return new_clauses


def preprocess(file_path):
    n_vars, clauses = parse_cnf_file(file_path)
    clauses = remove_dumplicate_literals(clauses)
    clauses, fixed_vars = unit_propagation(clauses)
    n_vars, clauses = remove_fixed_vars(n_vars, clauses, fixed_vars)
    clauses = add_unused_variables(n_vars, clauses)
    return n_vars, clauses


def VIG(n_vars, clauses):
    G = nx.Graph()
    G.add_nodes_from(range(n_vars))

    for clause in clauses:
        v_idxs = [literal2v_idx(literal)[1] for literal in clause]
        edges = list(combinations(v_idxs, 2))
        G.add_edges_from(edges)
    
    return G
