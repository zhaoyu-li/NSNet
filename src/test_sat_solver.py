import os
import sys
import argparse
import glob
import pickle
import time
from tqdm import tqdm

from utils.logger import Logger
from utils.solvers import SATSolver
from concurrent.futures.process import ProcessPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--solver', type=str, choices=['Sparrow'], default='Sparrow', help='Solver to be tested')
    parser.add_argument('--max_flips', type=int, default=None, help='Maximum number of flips in SLS Solver')
    parser.add_argument('--model', type=str, default=None, help='Model used for initialization')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')
    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')
    parser.add_argument('--trial', type=int, default=0, help='Experiment number')

    opts = parser.parse_args()

    opts.log_dir = os.path.join('runs', opts.solver)
    opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.eval_dir, exist_ok=True)

    dataset_name = os.path.abspath(opts.test_dir).split(os.path.sep)[-2]

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    print(opts)

    t0 = time.time()

    solver = SATSolver(opts)
    
    print('Testing...')
    all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        results = pool.map(solver.run, all_files)
    
    all_results = []
    tot = len(all_files)
    cnt = 0
    avg_flips = 0

    for result in results:
        all_results.append(result)
        complete, assignment, num_flips, t = result
        if complete:
            cnt += 1
            avg_flips += num_flips

    r = cnt / tot
    if cnt > 0:
        avg_flips /= cnt
    
    print('Total: %d, Solved: %d, Ratio: %.3f, Average number of flips: %.1f.' % (tot, cnt, r, avg_flips))

    t = time.time() - t0
    print('Solving Time: %f' % t)

    if opts.max_flips is None:
        opts.max_flips = 'inf'
    if opts.model is None:
        opts.model = 'none'

    with open('%s/dataset=%s_max_flips=%s_model=%s_timeout=%s_trial=%s.pkl' % \
        (opts.eval_dir, dataset_name, opts.max_flips, opts.model, opts.timeout, opts.trial), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
