import pickle
import argparse
import glob
import os
import math
import numpy as np

from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Directory with MC data')
    parser.add_argument('mc_eval_dir', type=str, help='Directory with MC result files')
    parser.add_argument('mis_eval_dir', type=str, help='Directory with MIS result files')
    parser.add_argument('--mc_timeout', type=int, default=5000, help='MC timeout')
    parser.add_argument('--mis_timeout', type=int, default=1000, help='MIS timeout')
    opts = parser.parse_args()

    all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))   
    dataset = os.path.abspath(opts.test_dir).split(os.path.sep)[-2]
    mc_result_file = os.path.join(opts.mc_eval_dir, 'dataset=%s_timeout=%s.pkl' % (dataset, opts.mc_timeout))
    mis_result_file = os.path.join(opts.mis_eval_dir, 'dataset=%s_timeout=%s.pkl' % (dataset, opts.mis_timeout))
    mis_mc_result_file = os.path.join(opts.mc_eval_dir, 'dataset=%s_MIS_timeout=%s.pkl' % (dataset, opts.mc_timeout))
    label_file_path = os.path.join(opts.test_dir, 'countings.pkl')

    rmse = defaultdict(int)
    cnt = defaultdict(int)

    tot_rmse = 0
    tot_runtime = 0
    tot_cnt = 0
    
    with open(mc_result_file, 'rb') as f:
        mc_results = pickle.load(f)
    
    with open(mis_result_file, 'rb') as f:
        mis_results = pickle.load(f)
    
    with open(mis_mc_result_file, 'rb') as f:
        mis_mc_results = pickle.load(f)
    
    with open(label_file_path, 'rb') as f:
        labels = pickle.load(f)

    for f, mc_result, mis_result, mc_mis_result, label in zip(all_files, mc_results, mis_results, mis_mc_results, labels):
        mc_complete, mc_counting, mc_t = mc_result
        mis_complete, ind_vars, mis_t = mis_result
        mc_mis_complete, mc_mis_counting, mc_mis_t = mc_mis_result

        complete = mc_complete
        if complete:
            ln_counting = float(mc_counting.ln())
            t = mc_t

        if mis_complete and mc_mis_complete:
            if not complete or mis_t + mc_mis_t < mc_t:
                complete = 1
                ln_counting = float(mc_mis_counting.ln())
                t = mis_t + mc_mis_t

        if complete:
            category = os.path.abspath(f).split(os.path.sep)[-2]
            rmse[category] += (ln_counting - label)**2
            cnt[category] += 1
            tot_rmse += (ln_counting - label)**2
            tot_cnt += 1
            tot_runtime += t
    
    for category in rmse.keys():
        rmse[category] = math.sqrt(rmse[category] / cnt[category])
        print('%s, RMSE: %.2f' % (category, rmse[category]))
    
    print('total RMSE: %.2f, avg. runtime: %.2f' % (math.sqrt(tot_rmse / tot_cnt), tot_runtime / tot_cnt))


if __name__ == '__main__':
    main()