import os
import sys
import argparse
import glob
import math
import pickle
import shutil

from utils.logger import Logger
from utils.solvers import MISSolver
from concurrent.futures.process import ProcessPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('--solver', type=str, default='MIS', help='Solver to be used')
    parser.add_argument('--timeout', type=int, default=1000, help='Timeout')
    parser.add_argument('--n_process', type=int, default=32, help='Number of processes to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    opts = parser.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)

    opts.log_dir = os.path.join('runs', 'MIS')
    opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.eval_dir, exist_ok=True)

    dataset_name = os.path.abspath(opts.test_dir).split(os.path.sep)[-2]

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    print(opts)

    labels_file = os.path.join(opts.test_dir, 'countings.pkl')
    shutil.copyfile(labels_file, os.path.join(opts.out_dir, 'countings.pkl'))

    solver = MISSolver(opts)
    
    print('Generating...')
    all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        results = pool.map(solver.run, all_files)
    
    all_results = []
    cnt = 0
    for file_path, result in zip(all_files, results):
        all_results.append(result)
        complete, ind_vars, t = result
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if complete:
            ind_line = 'c ind ' + ' '.join([str(var) for var in ind_vars]) + ' 0\n'
            lines.insert(0, ind_line)
        
        out_path = os.path.join(opts.out_dir, '%.5d.cnf' % (cnt))
        with open(out_path, 'w') as f:
            f.writelines(lines)
        
        cnt += 1

    with open('%s/dataset=%s_timeout=%s.pkl' % (opts.eval_dir, dataset_name, opts.timeout), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()
