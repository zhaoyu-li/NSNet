import os
import sys
import argparse
import glob
import math
import pickle

from utils.logger import Logger
from utils.solvers import MCSolver
from concurrent.futures.process import ProcessPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--solver', type=str, choices=['ApproxMC3', 'F2', 'DSHARP'], default='ApproxMC', help='Solver to be tested')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')
    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

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

    solver = MCSolver(opts)
    
    print('Testing...')
    all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        results = pool.map(solver.run, all_files)

    all_results = []
    tot = len(all_files)
    failed = 0
    rmse = 0

    labels_file = os.path.join(opts.test_dir, 'countings.pkl')
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    for result, label in zip(results, labels):
        all_results.append(result)
        complete, counting, t = result
        if complete:
            ln_counting = float(counting.ln())
            rmse += (ln_counting - label) ** 2
        else:
            failed += 1
    
    rmse = math.sqrt(rmse / (tot - failed))
    print('Total: %d, Failed: %d, RMSE: %f.' % (tot, failed, rmse))

    with open('%s/dataset=%s_timeout=%s.pkl' % (opts.eval_dir, dataset_name, opts.timeout), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
