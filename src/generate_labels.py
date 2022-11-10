import os
import argparse
import glob
import pickle
import shutil

from utils.solvers import MCSolver, SATSolver, MESolver
from concurrent.futures.process import ProcessPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['model-counting', 'assignment', 'marginal'], help='Task')
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    parser.add_argument('--out_dir', type=str, default=None, help='Output Directory with sat data')
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')

    opts = parser.parse_args()
    print(opts)

    if opts.task == 'model-counting':
        opts.solver = 'DSHARP'
    elif opts.task == 'assignment':
        opts.solver = 'CaDiCaL'
    else:
        opts.solver = 'bdd_minisat_all'

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)

    if opts.task == 'model-counting':
        solver = MCSolver(opts)
    elif opts.task == 'assignment':
        solver = SATSolver(opts)
    else:
        solver = MESolver(opts)

    labels = []
    
    print('Generating labels...')
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        results = pool.map(solver.run, all_files)
    
    tot = len(all_files)
    cnt = 0

    for i, result in enumerate(results):
        if opts.task == 'model-counting':
            complete, counting, t = result
        elif opts.task == 'assignment':
            complete, assignment, _, t = result
        else:
            complete, marginal, t = result
        
        if complete:
            cnt += 1
            if opts.task == 'model-counting':
                ln_counting = float(counting.ln())
                labels.append(ln_counting)
            elif opts.task == 'assignment':
                labels.append(assignment)
            else:
                labels.append(marginal)
            
            if opts.out_dir is not None:
                shutil.copyfile(all_files[i], os.path.join(opts.out_dir, '%.5d.cnf' % (cnt)))
        else:
            if opts.out_dir is None:
                os.remove(all_files[i])

    r = cnt / tot
    print('Total: %d, Labeled: %d, Ratio: %.4f.' % (tot, cnt, r))

    if opts.out_dir is not None:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.out_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.out_dir, 'assignments.pkl')
        else:
            labels_file = os.path.join(opts.out_dir, 'marginals.pkl')
    else:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.data_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.data_dir, 'assignments.pkl')
        else:
            labels_file = os.path.join(opts.data_dir, 'marginals.pkl')
    
    with open(labels_file, 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    main()
