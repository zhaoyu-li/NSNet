import os
import argparse
import glob
import shutil
import numpy as np
import pickle

from itertools import accumulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with SAT data')
    parser.add_argument('--keep_category', action='store_true', help='Keep the category directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    opts = parser.parse_args()
    print(opts)

    print('Splitting data ...')
    
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True)) # sorted by category and then file names
    all_files = [os.path.abspath(f) for f in all_files]

    frac_list = np.asarray([0.6, 0.2, 0.2])
    
    num_data = len(all_files)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])

    indices = np.random.RandomState(seed=opts.seed).permutation(num_data)

    split_indices = [indices[offset-length:offset] for offset, length in zip(accumulate(lengths), lengths)]

    split_idx = {
        'train': sorted(split_indices[0]),
        'valid': sorted(split_indices[1]),
        'test': sorted(split_indices[2])
    }

    train_dir = os.path.join(opts.data_dir, 'train')
    valid_dir = os.path.join(opts.data_dir, 'valid')
    test_dir = os.path.join(opts.data_dir, 'test')

    categories = [category for category in os.listdir(opts.data_dir) \
        if os.path.isdir(os.path.join(opts.data_dir, category))]
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    countings_file = os.path.join(opts.data_dir, 'countings.pkl')
    if os.path.exists(countings_file):
        train_countings_file = os.path.join(train_dir, 'countings.pkl')
        valid_countings_file = os.path.join(valid_dir, 'countings.pkl')
        test_countings_file = os.path.join(test_dir, 'countings.pkl')
        
        with open(countings_file, 'rb') as f:
            countings = pickle.load(f)
        train_countings = [countings[idx] for idx in split_idx['train']]
        valid_countings = [countings[idx] for idx in split_idx['valid']]
        test_countings = [countings[idx] for idx in split_idx['test']]
        
        with open(train_countings_file, 'wb') as f:
            pickle.dump(train_countings, f)
        with open(valid_countings_file, 'wb') as f:
            pickle.dump(valid_countings, f)
        with open(test_countings_file, 'wb') as f:
            pickle.dump(test_countings, f)
        
        os.remove(countings_file)
    
    assignments_file = os.path.join(opts.data_dir, 'assignments.pkl')
    if os.path.exists(assignments_file):
        train_assignments_file = os.path.join(train_dir, 'assignments.pkl')
        valid_assignments_file = os.path.join(valid_dir, 'assignments.pkl')
        test_assignments_file = os.path.join(test_dir, 'assignments.pkl')
        
        with open(assignments_file, 'rb') as f:
            assignments = pickle.load(f)
        train_assignments = [assignments[idx] for idx in split_idx['train']]
        valid_assignments = [assignments[idx] for idx in split_idx['valid']]
        test_assignments = [assignments[idx] for idx in split_idx['test']]
        
        with open(train_assignments_file, 'wb') as f:
            pickle.dump(train_assignments, f)
        with open(valid_assignments_file, 'wb') as f:
            pickle.dump(valid_assignments, f)
        with open(test_assignments_file, 'wb') as f:
            pickle.dump(test_assignments, f)
        
        os.remove(assignments_file)
    
    if opts.keep_category:
        for category in categories:
            train_category_dir = os.path.join(train_dir, category)
            valid_category_dir = os.path.join(valid_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(valid_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)

    for t, idx in enumerate(split_idx['train']):
        if opts.keep_category:
            file_path = all_files[idx]
            dir_names = file_path.replace(opts.data_dir, '').split(os.path.sep)
            dir_names = [dir_name for dir_name in dir_names if dir_name]
            category = dir_names[0]
            train_category_dir = os.path.join(train_dir, category)
            shutil.move(file_path, os.path.join(train_category_dir, '%.5d.cnf' % t))
        else:
            file_path = all_files[idx]
            shutil.move(file_path, os.path.join(train_dir, '%.5d.cnf' % t))

    for t, idx in enumerate(split_idx['valid']):
        if opts.keep_category:
            file_path = all_files[idx]
            dir_names = file_path.replace(opts.data_dir, '').split(os.path.sep)
            dir_names = [dir_name for dir_name in dir_names if dir_name]
            category = dir_names[0]
            valid_category_dir = os.path.join(valid_dir, category)
            shutil.move(file_path, os.path.join(valid_category_dir, '%.5d.cnf' % t))
        else:
            file_path = all_files[idx]
            shutil.move(file_path, os.path.join(valid_dir, '%.5d.cnf' % t))

    for t, idx in enumerate(split_idx['test']):
        if opts.keep_category:
            file_path = all_files[idx]
            dir_names = file_path.replace(opts.data_dir, '').split(os.path.sep)
            dir_names = [dir_name for dir_name in dir_names if dir_name]
            category = dir_names[0]
            test_category_dir = os.path.join(test_dir, category)
            shutil.move(file_path, os.path.join(test_category_dir, '%.5d.cnf' % t))
        else:
            file_path = all_files[idx]
            shutil.move(file_path, os.path.join(test_dir, '%.5d.cnf' % t))
    
    for category in categories:
        unused_dir = os.path.join(opts.data_dir, category)
        shutil.rmtree(unused_dir)


if __name__ == '__main__':
    main()
