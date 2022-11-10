import os
import argparse
import random
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from cnfgen import RandomKCNF
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
            n_clauses = int(4.258 * n_vars + 58.26 * pow(n_vars, -2 / 3.))
            
            cnf = RandomKCNF(3, n_vars, n_clauses)
            clauses = list(cnf.clauses())
            clauses = [list(cnf._compress_clause(clause)) for clause in clauses]
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                write_dimacs_to(n_vars, clauses, os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (t))))
                break
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=100)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))


if __name__ == '__main__':
    main()