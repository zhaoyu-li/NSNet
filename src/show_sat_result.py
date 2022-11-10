import pickle
import argparse
import glob
import os
import math
import numpy as np

from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir', type=str, help='Directory with result files')
    parser.add_argument('datasets', type=str, nargs='+', help='Dataset name')
    parser.add_argument('--max_flips', type=int, default=100, help='Maximum number of flips in SLS Solver')
    parser.add_argument('--model', type=str, default=None, help='Model used for initialization')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')
    opts = parser.parse_args()

    if opts.max_flips is None:
        opts.max_flips = 'inf'
    if opts.model is None:
        opts.model = 'none'

    for dataset in opts.datasets:
        file_name = 'dataset=%s_max_flips=%s_model=%s_timeout=%s_trial=*.pkl' % \
            (dataset, opts.max_flips, opts.model, opts.timeout)

        result_files = sorted(glob.glob(os.path.join(opts.eval_dir, file_name), recursive=True))

        if not result_files:
            continue

        acc_results = []
        flips_results = []
        
        for i, result_file in enumerate(result_files):
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            solved = 0
            flips = 0
            cnt = len(results)

            for result in results:
                complete, assignment, num_flips, t = result

                if complete:
                    solved += 1
                    flips += num_flips
            
            acc = solved / cnt
            avg_flips = flips / solved
            acc_results.append(acc)
            flips_results.append(avg_flips)
        
        print(dataset, 'Acc: %.2f%%(%.2f%%), #Flips: %.2f(%.2f)' % (np.mean(acc_results) * 100, np.std(acc_results) * 100, np.mean(flips_results), np.std(flips_results)))


if __name__ == '__main__':
    main()