import os
import re
import signal
import time
import subprocess
import numpy as np
import pickle

from decimal import Decimal


class SATSolver:
    def __init__(self, opts):
        self.opts = opts
        if opts.solver == 'CaDiCaL':
            self.exec_dir = os.path.abspath('external/CaDiCaL')
            self.cmd_line = ['./cadical']
        elif opts.solver == 'Sparrow':
            self.exec_dir = os.path.abspath('external/Sparrow')
            self.cmd_line = ['./sparrow', '-a', '-l', '-r1']
            if opts.max_flips is not None:
                self.cmd_line.append('--maxflips')
                self.cmd_line.append(str(opts.max_flips))

    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]        
        cmd_line = self.cmd_line.copy()
        if self.opts.solver == 'Sparrow' and self.opts.model is not None:
            tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '_' + self.opts.model + '.out')
            init_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.model + '.out')
            cmd_line.append('-f')
            cmd_line.append(init_filepath)
        else:
            tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        
        cmd_line.append(input_filepath)

        with open(tmp_filepath, 'w') as f:
            t0 = time.time()
            timeout_expired = 0
            try:
                process = subprocess.Popen(cmd_line, stdout=f, stderr=f, cwd=self.exec_dir, start_new_session=True)
                process.communicate(timeout=self.opts.timeout)
                # may also finished by linux oom killer
            except:
                timeout_expired = 1
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            t = time.time() - t0
        
        complete = 0
        assignment = []
        num_flips = 0

        if timeout_expired or os.stat(tmp_filepath).st_size == 0: # timeout
            os.remove(tmp_filepath)
            return complete, assignment, num_flips, t
        
        with open(tmp_filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('v'):
                    assignment = assignment + [int(s) for s in line.strip().split()[1:]]
                if line.startswith('c numFlips'): # Local search solver
                    num_flips = Decimal(line.strip().split()[-1])
        
        if assignment: # All instances are SAT
            complete = 1
            assignment = np.array(assignment[:-1]) > 0
        
        os.remove(tmp_filepath)
        return complete, assignment, num_flips, t


class MCSolver:
    def __init__(self, opts):
        self.opts = opts
        if opts.solver == 'DSHARP':
            self.exec_dir = os.path.abspath('external/DSHARP')
            self.cmd_line = ['./dsharp']
            # with gmp
            self.cnt_pattern = '#SAT \(full\):   \t\t(.+)\n'
            # without gmp
            # self.cnt_pattern = '# of solutions:\t\t(.+)\n'
        elif opts.solver == 'ApproxMC3':
            self.exec_dir = os.path.abspath('external/ApproxMC3')
            self.cmd_line = ['./approxmc3']
            self.cnt_pattern = 'Number of solutions is: (.+)\n'
            # approxmc4
            # self.cnt_pattern = 's mc (.+)\n'
        elif opts.solver == 'F2':
            self.exec_dir = os.path.abspath('external/F2')
            self.cmd_line = ['python', 'f2.py', '--random-seed', str(abs(opts.seed)+1), '--sharpsat-exe', 'sharpsat', \
                '--mode', 'lb', '--max-time', str(opts.timeout), '--skip-sharpsat']
            self.cnt_pattern = 'F2: Lower bound is (.+) \('

    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        stdout = ''

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, cwd=self.exec_dir, text=True, start_new_session=True)
            stdout, _ = process.communicate(timeout=self.opts.timeout)
            # may also finished by linux oom killer
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0
        
        complete = 0
        counting = -1

        matches = re.search(self.cnt_pattern, stdout)

        if self.opts.solver == 'F2': # remove tmp files if F2 doesn't remove them (we modify the tmp filenames for F2)
            all_files = os.listdir(self.exec_dir)
            for tmp_file in all_files:
                if tmp_file.endswith('.cnf') and filename in tmp_file:
                    os.remove(os.path.join(self.exec_dir, tmp_file))

        if timeout_expired or not matches:
            return complete, counting, t

        complete = 1
        if 'x' not in matches[1] and '^' not in matches[1]:
            counting = Decimal(matches[1])
            if Decimal.is_nan(counting):
                complete = 0
        else:
            counting = Decimal(eval(matches[1].replace('x', '*').replace('^', '**')))
            if Decimal.is_nan(counting):
                complete = 0
        
        return complete, counting, t


class MISSolver:
    def __init__(self, opts):
        self.opts = opts
        assert self.opts.solver == 'MIS'
        self.exec_dir = os.path.abspath('external/MIS')
        self.cmd_line = ['python', 'mis.py']
    
    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        cmd_line.append('--out')
        cmd_line.append(tmp_filepath)

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate(timeout=self.opts.timeout)
            # may also finished by linux oom killer
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0

        complete = 0
        ind_vars = None

        if timeout_expired or not os.path.exists(tmp_filepath):
            return complete, ind_vars, t
        
        complete = 1
        with open(tmp_filepath, 'r') as f:
            lines = f.readlines()
            # assert len(lines) == 1
            ind_vars = [int(s) for s in lines[0].strip().split()[:-1]]
        
        os.remove(tmp_filepath)
        return complete, ind_vars, t


class MESolver:
    def __init__(self, opts):
        self.opts = opts
        assert self.opts.solver == 'bdd_minisat_all'
        self.exec_dir = os.path.abspath('external/bdd_minisat_all')
        self.cmd_line = ['python', 'bdd_minisat_all.py']
    
    def run(self, input_filepath):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        tmp_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.out')
        output_filepath = os.path.join(os.path.dirname(input_filepath), filename + '_' + self.opts.solver + '.pkl')
        cmd_line = self.cmd_line.copy()
        cmd_line.append(input_filepath)
        cmd_line.append(tmp_filepath)
        cmd_line.append(output_filepath)

        t0 = time.time()
        timeout_expired = 0
        try:
            process = subprocess.Popen(cmd_line, cwd=self.exec_dir, start_new_session=True)
            process.communicate(timeout=self.opts.timeout)
            # may also finished by linux oom killer
        except:
            timeout_expired = 1
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        t = time.time() - t0

        complete = 0
        marginal = None

        if timeout_expired or not os.path.exists(output_filepath):
            os.remove(tmp_filepath)
            return complete, marginal, t
        
        complete = 1
        with open(output_filepath, 'rb') as f:
            marginal = pickle.load(f)
        
        os.remove(tmp_filepath)
        os.remove(output_filepath)

        return complete, marginal, t
