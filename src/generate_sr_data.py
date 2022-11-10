import os
import argparse
import numpy as np
import random
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from utils.utils import write_dimacs_to, VIG


class Generator:
    def __init__(self, opts):
        self.opts = opts
        os.makedirs(self.opts.out_dir, exist_ok=True)

    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)

        while True:
            n_vars = random.randint(self.opts.min_n, self.opts.max_n)
            solver = Cadical()
            clauses = []
            while True:
                k_base = 1 if random.random() < self.opts.p_k_2 else 2
                k = k_base + np.random.geometric(self.opts.p_geo)
                k = 4 if k > 4 else k
                vs = np.random.choice(n_vars, size=min(n_vars, k), replace=False)
                clause = [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

                solver.add_clause(clause)
                if solver.solve():
                    clauses.append(clause)
                else:
                    break
            
            sat_clause = [-clause[0]] + clause[1:]
            clauses.append(sat_clause)

            vig = VIG(n_vars, clauses)
            if nx.is_connected(vig):
                break

        write_dimacs_to(n_vars, clauses, os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (t))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=100)

    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))
    

if __name__ == '__main__':
    main()
