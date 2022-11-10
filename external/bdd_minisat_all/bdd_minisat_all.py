import os
import argparse
import subprocess
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('tmp_output_file', type=str, help='Temporary output file')
    parser.add_argument('output_file', type=str, help='Output file')
    opts = parser.parse_args()
    
    cmd_line = ['./bdd_minisat_all', opts.input_file, opts.tmp_output_file]
    
    subprocess.run(cmd_line) # may also finished by linux oom killer

    with open(opts.tmp_output_file, 'r') as f: # may also finished by linux oom killer
        lines = f.readlines()
        counting = len(lines)
        n_vars = len(lines[0].strip().split()) - 1
        marginal = np.zeros(n_vars)
        for line in lines:
            assignment = np.array([int(s) for s in line.strip().split()[:-1]])
            assignment = assignment > 0
            marginal += assignment
        marginal /= counting

    with open(opts.output_file, 'wb') as f:
        pickle.dump(marginal, f)


if __name__ == '__main__':
    main()