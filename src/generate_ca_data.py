import os
import argparse
import random
import subprocess
import networkx as nx

from concurrent.futures.process import ProcessPoolExecutor
from pysat.solvers import Cadical
from utils.utils import parse_cnf_file, VIG


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.exec_dir = os.path.abspath('external/CA')
        os.makedirs(self.opts.out_dir, exist_ok=True)
        
    def run(self, t):
        if t % self.opts.print_interval == 0:
            print('Generating instance %d.' % t)
        
        while True:
            n_vars = random.randint(self.opts.min_n, self.opts.max_n)
            r = random.uniform(3, 4)
            n_clauses = int(r * n_vars)
            n_communities = min(random.randint(self.opts.min_c, self.opts.max_c), n_vars/3)
            modularity = random.uniform(self.opts.min_q, self.opts.max_q)
            
            cnf_filepath = os.path.abspath(os.path.join(self.opts.out_dir, '%.5d.cnf' % (t)))
            cmd_line = ['./ca', '-n', str(n_vars), '-m', str(n_clauses), '-c', str(n_communities), \
                '-Q', str(modularity), '-s', str(random.randint(0, 2**32)), '-o', cnf_filepath]
            
            try:
                process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
                process.communicate()
            except:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            if not os.path.exists(cnf_filepath):
                continue
            
            n_vars, clauses = parse_cnf_file(cnf_filepath)
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                os.remove(cnf_filepath)
                continue

            solver = Cadical(bootstrap_with=clauses)
            
            if solver.solve():
                break
            
            os.remove(cnf_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str)
    parser.add_argument('n_instances', type=int)

    parser.add_argument('--min_n', type=int, default=10)
    parser.add_argument('--max_n', type=int, default=40)

    parser.add_argument('--min_c', type=int, default=3)
    parser.add_argument('--max_c', type=int, default=10)
    
    parser.add_argument('--min_q', type=float, default=0.7)
    parser.add_argument('--max_q', type=float, default=0.9)

    parser.add_argument('--print_interval', type=int, default=1000)

    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')

    opts = parser.parse_args()

    generater = Generator(opts)
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        pool.map(generater.run, range(opts.n_instances))


if __name__ == '__main__':
    main()
