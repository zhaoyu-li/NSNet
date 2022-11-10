#!/usr/bin/env python3
# Copyright (c) 2018, Panos Theodoropoulos

import argparse
import time
import math
import random
import os
import os.path
import shlex
import subprocess
import networkx as nx
from networkx.algorithms import bipartite
import sympy as sp
import scipy.optimize
import logging as log


class Config:
    def __init__(self):
        self.formula_filename = None
        self.has_ind_header = None
        self.var_set = None  # list of variables used in the constraints
        self.working_dir = None
        self.keep_cnf = False
        self.skip_sharpsat = False
        self.random_seed = None
        self.mode = None
        self.default_error_prob = 0.05
        self.total_max_time = math.nan
        # self.f2_program_git_version = ''  # DEVTODO: Autocomplete
        # self.cmsat_git_version = ''  # DEVTODO: Autocomplete
        self.cmsat_major_version = 5

        # Preferred time allocated for each solver invocation, actual may be
        # larger (if iterations are few) or smaller (if total time is being
        # exhausted)
        self.alg1_loop_timeout = None
        self.degn = math.nan  # Average variable degree of LDPC code
        self.alg1_maxsol = 4
        self.alg1_threshold_coeff = 2
        self.alg1_n_iter = math.nan
        self.alg2_n_iter = math.nan
        self.alg2_loop_timeout = math.nan
        self.alg2_maxsol = 999999999
        self.alg2_error_prob = math.nan
        self.f2_alg_confidence_radius = math.nan
        self.f2_alg_error_prob = math.nan
        self.args = None
        self.cmsat_exe = None
        self.sharpsat_exe = None

    def setup(self, args):
        global formula_lines
        self.mode = args.mode
        if not os.path.isfile(args.input_filename):
            log.error('File {} does not exist.'.format(args.input_filename))
            exit(101)
        else:
            self.formula_filename = args.input_filename
            with open(args.input_filename, 'r') as f:
                formula_lines = f.readlines()
                self.var_set, self.has_ind_header = \
                    extract_support_vars()
        if args.working_dir == '':
            conf.working_dir = os.getcwd()
        elif not os.path.isdir(args.working_dir):
            log.error('Working directory does not exist.')
            exit(102)
        else:
            conf.working_dir = args.working_dir
            os.chdir(conf.working_dir)
        self.keep_cnf = args.keep_cnf
        self.skip_sharpsat = args.skip_sharpsat
        if args.random_seed >= 0:
            self.random_seed = args.random_seed
            random.seed(args.random_seed)
        else:
            random.seed()
            rs = random.randint(0, 2**31)
            random.seed(rs)
            self.random_seed = rs
        if (not math.isnan(args.lower_bound)) and\
                (not math.isnan(args.upper_bound)):
            if args.lower_bound >= math.isnan(args.upper_bound):
                log.error('lower_bound must be less than upper_bound')
                exit(103)
        self.degn = args.var_degree
        self.cmsat_major_version = args.cmsat_version
        self.total_max_time = args.max_time
        if args.cmsat_exe is not None:
            self.cmsat_exe = os.path.join('.', args.cmsat_exe)
            if not os.path.isfile(self.cmsat_exe):
                log.error('CMSat executable does not exist')
                exit(104)
        else:
            f2_dir = os.path.dirname(os.path.realpath(__file__))
            cms = os.path.join(f2_dir, 'cryptominisat5')
            if not os.path.isfile(cms):
                log.error('CMSat executable does not exist in F2 dir')
                exit(105)
            else:
                self.cmsat_exe = cms
        if args.sharpsat_exe is not None:
            self.sharpsat_exe = os.path.join('.', args.sharpsat_exe)
            if not os.path.isfile(self.sharpsat_exe):
                log.error('sharpsat executable does not exist')
                exit(108)
        else:
            f2_dir = os.path.dirname(os.path.realpath(__file__))
            shs = os.path.join(f2_dir, 'sharpsat')
            if not os.path.isfile(shs):
                log.error('sharpSAT executable does not exist in F2 dir')
                exit(109)
            else:
                self.sharpsat_exe = shs
        if math.isnan(args.error_prob):
            error_probability = conf.default_error_prob
        else:
            error_probability = args.error_prob
        if args.mode == 'lb':
            lb_mode_sanity_check(args)
            self.setup_alg1(args, error_probability)
        elif args.mode == 'ub':
            ub_mode_sanity_check(args)
            if not math.isnan(args.lower_bound):
                self.setup_alg2(args, error_probability)
            else:  # Lower bound not given
                if not math.isnan(args.lb_n_iter):
                    self.setup_alg1(args, math.nan)
                    self.setup_alg2(args, error_probability)
                else:
                    self.setup_alg1(args, error_probability / 2)
                    self.setup_alg2(args, error_probability / 2)
        elif args.mode == 'appr':
            num_of_phases = 1  # In how many parts to split the total error pr.
            if not math.isnan(args.lower_bound):
                num_of_phases += 1
            if not math.isnan(args.upper_bound):
                num_of_phases += 1
            er_prob = error_probability / num_of_phases
            if math.isnan(args.lower_bound):
                self.setup_alg1(args, er_prob)
            if math.isnan(args.upper_bound):
                self.setup_alg2(args, er_prob)
            self.setup_f2_alg(args, er_prob)
        else:
            assert False

    def setup_alg2(self, args, er_prob):
        """
        Setup the configuration for algorithm 2.
        If number of iterations is given, it is used. Otherwise it will be
        calculated from the available 'share' of error probability given as
        second parameter inside the algorithm
        :param args:
        :param er_prob: Error probability for the invocation of algorithm
        """
        if not math.isnan(args.ub_n_iter):
            self.alg2_n_iter = args.ub_n_iter
        else:
            self.alg2_error_prob = er_prob
        self.alg2_loop_timeout = args.max_time / min(5, self.alg2_n_iter)

    def setup_alg1(self, args, er_prob):
        """
        Setup the configuration for algorithm 1.
        If number of iterations is given, it is used. Otherwise it is
        calculated from the available 'share' of error probability given as
        second parameter.
        :param args:
        :param er_prob: Error probability for the invocation of algorithm
        """
        if not math.isnan(args.lb_n_iter):
            self.alg1_n_iter = args.lb_n_iter
        else:
            self.alg1_n_iter = int(math.ceil(-8 * math.log(er_prob)))
        self.alg1_loop_timeout = args.max_time / min(5, self.alg1_n_iter)

    def setup_f2_alg(self, args, er_prob):
        self.f2_alg_confidence_radius = args.confidence_radius
        self.f2_alg_error_prob = er_prob


# === Globals ===
conf = Config()  # Global configuration of all algorithms
formula_lines = []  # The lines of the original cnf file
start_time = time.process_time()
aug_counter = 0  # Counter for augmented formulas filenames
subprocess_cumulative_time = 0  # Accumulate the time spent in subprocesses
# ===============


def lb_mode_sanity_check(args):
    if (not math.isnan(args.lb_n_iter) and
            not math.isnan(args.error_prob)):
        log.error('In \'lb\' mode --error_prob and --lb_n_iter '
                  'cannot be both specified.')
        exit(107)
    if math.isnan(args.lb_n_iter) and math.isnan(args.error_prob):
        args.error_prob = 0.05


def ub_mode_sanity_check(args):
    if (not math.isnan(args.ub_n_iter) and
            not math.isnan(args.error_prob)):
        log.error('In \'ub\' mode --error_prob and --ub_n_iter '
                  'cannot be both specified.')
        exit(106)
    if math.isnan(args.ub_n_iter) and math.isnan(args.error_prob):
        args.error_prob = 0.05


def check_time():
    this_process_time = time.process_time() - start_time
    rt = conf.total_max_time - this_process_time - subprocess_cumulative_time
    if rt <= 0:
        raise TotalTimeoutException()


def remaining_time():
    """
    Calculate the remaining time allowed for the program to run
    """
    this_process_time = time.process_time() - start_time
    rt = conf.total_max_time - this_process_time - subprocess_cumulative_time
    return max(rt, 1)


def consumed_time():
    """
    Total time spent by this process the the subprocesses it spawned
    :return: time in seconds
    """
    this_process_time = time.process_time() - start_time
    return this_process_time + subprocess_cumulative_time


class SolverBoundsExceededException(Exception):
    """
    A solver timeout occurred or the number of solutions exceeded a bound
    in a case that it is fatal for the algorithm.
    """
    pass


class TotalTimeoutException(Exception):
    """
    The total time allowed running time has expired.
    """
    pass


def main():
    global conf
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        prog='F2',
        usage='%(prog)s [options] formula_file.cnf',
        description=''
        'Probabilistically approximate number of models of a cnf formula')
    parser.add_argument('input_filename', type=str,
                        help='The formula in extended DIMACS form')
    parser.add_argument('--working-dir', type=str,
                        help='Where to create auxiliary files', default='')
    parser.add_argument('--keep-cnf', action='store_true',
                        help='Keep generated auxiliary formulas?')
    parser.add_argument('--random-seed', type=int,
                        help='Initialize the random generator', default=42)
    parser.add_argument('--var-degree', type=int,
                        help='Average variable degree of LDPC XORs',
                        default=12)
    parser.add_argument('--cmsat-exe', type=str,
                        help='The location of a cryptominisat executable',
                        default=None)
    parser.add_argument('--cmsat-version', type=int,
                        help='The major version of cryptominisat executable. '
                             '(Allowed versions 2 or 5, default 5)',
                        default=5)
    parser.add_argument('--sharpsat-exe', type=str,
                        help='The location of a SharpSAT executable',
                        default=None)
    parser.add_argument('--mode', type=str,
                        help='Mode of operation. Allowed values: lb, ub, appr',
                        choices=['lb', 'ub', 'appr'],
                        default='appr')
    parser.add_argument('--lower-bound', type=int,
                        help='Binary logarithm of lower bound (modes: ub, '
                             'appr)', default=math.nan)
    parser.add_argument('--upper-bound', type=int,
                        help='Binary logarithm of upper bound (only appr '
                             'mode)', default=math.nan)
    parser.add_argument('--error-prob', type=float,
                        help='Probability of error', default=math.nan)
    parser.add_argument('--confidence-radius', type=float,
                        help='Tolerance of error as a ratio of real value',
                        default=0.2)
    parser.add_argument('--max-time', type=int,
                        help='Maximum system running time (in seconds)',
                        default=3600)
    parser.add_argument('--lb-n-iter', type=int,
                        help='Override num of iterations in algorithm 1',
                        default=math.nan)
    parser.add_argument('--ub-n-iter', type=int,
                        help='Override num of iterations in algorithm 2',
                        default=math.nan)
    parser.add_argument('--skip-sharpsat', action='store_true',
                        help='Skip the SharpSAT invocation.')
    args = parser.parse_args()
    conf.setup(args)
    run(args)


def run(args):
    try:
        if not conf.skip_sharpsat:
            log.info('Running SharpSAT for quick test...')
            tout, n, ct = sharpsat_count(conf.formula_filename, 2)
            if not tout:
                if n > 1:
                    print('F2: Exact count (by SharpSAT) is {} '
                          '(t={:.2f})'.format(n, ct))
                    print('F2Sharp:{}:{:.2f}:{:.2f}'.format(
                        conf.formula_filename, math.log2(n), ct))
                    if conf.has_ind_header:
                        print_sharpsat_warning()
                    exit(0)
                elif n == 0:
                    log.info(
                        'F2: The formula is UNSAT (as reported by sharpSAT)')
                    print('F2: UNSAT')
                    print('F2Sharp:{}:{:.2f}:{:.2f}'.format(
                        conf.formula_filename, 0, ct))
                    exit(0)
                else:  # n == 1
                    log.info('SharpSAT returned 1 which means that the formula'
                             ' is either UNSAT or has 1 solution.')
                    if is_unsat():
                        print('F2: UNSAT')
                        exit(0)
            else:
                log.info('SharpSAT timed out. Proceeding...')
        if conf.mode == 'lb':
            lb_init = prepare_lower_bound()
            lb = find_lower_bound(lb_init)
            pt = consumed_time()
            print('F2: Lower bound is 2^{} (processor time: {:.2f})'
                  ''.format(lb, pt))
        elif conf.mode == 'ub':
            if math.isnan(args.lower_bound):
                lb_init = prepare_lower_bound()
                lb = find_lower_bound(lb_init)
                ct1 = consumed_time() - 2  # Remove the sharpsat time
            else:
                lb = args.lower_bound
                ct1 = 0
            ub_a, ub_b = find_upper_bound(lb)
            ct2 = consumed_time()
            print('F2: Lower bound is 2^{} (t={:.2f}) and upper bound '
                  'is {:.2f} * 2^{:.2f} (total processor time: {:.2f})'
                  ''.format(lb, ct1, ub_a, ub_b, ct2))
            print('F2UB:{}:{}:{:.2f}:{:.2f}:{:.2f}'.format(
                  conf.formula_filename, lb, ct1,
                  math.log2(ub_a) + ub_b, ct2 - ct1))
        elif conf.mode == 'appr':
            if math.isnan(args.lower_bound):
                lb_init = prepare_lower_bound()
                lb = find_lower_bound(lb_init)
            else:
                lb = args.lower_bound
            if math.isnan(args.upper_bound):
                ub_a, ub_b = find_upper_bound(lb)
                ub = math.log2(ub_a) + ub_b
            else:
                ub = args.upper_bound
            if lb > ub:
                log.error('F2: Lower bound estimate cannot be greater than'
                          ' upper')
                exit(202)
            try:
                a, b = F2_algorithm(lb, ub, conf.f2_alg_confidence_radius,
                                    conf.f2_alg_error_prob)
                if a != -1:
                    pt = consumed_time()
                    print('F2: Z = {} * 2^{}  (processor time: {:.2f})'
                          ''.format(a, b, pt))
                else:
                    print('F2: Approximation Algorithm Failed!')
                    exit(203)
            except SolverBoundsExceededException:
                print('F2: Approximation Algorithm Failed!')
                exit(203)
        else:
            assert False, 'Unknown mode: {}'.format(conf.mode)
    except TotalTimeoutException:
        log.error('Total Timeout occured!')
        print('F2: The allowed running time expired before finish. Exiting.')
        exit(204)
    exit(0)


def prepare_lower_bound():
    # few_solutions = 2**4
    # if len(conf.var_set) <= 10:  # Take care of really "small" cases
    #     t, b, n, ctime = sat_count(conf.formula_filename,
    #                                conf.total_max_time - 2, few_solutions)
    #     if t:
    #         raise TotalTimeoutException
    #     elif n == 0:
    #         print('F2: UNSAT')
    #         exit(0)
    #     elif n < few_solutions:
    #         print('F2: The formula has exactly {} models'.format(n))
    #         exit(0)
    #     else:
    #         return math.log2(few_solutions)
    # else:
    if is_unsat():
        print('F2: UNSAT')
        exit(0)
    else:
        return 0


def is_unsat():
    t, b, n, ctime = sat_count(conf.formula_filename,
                               conf.total_max_time - 2, 1)
    if t:
        raise TotalTimeoutException
    elif b:
        return False
    else:
        assert (n == 0)
        return True


def find_lower_bound(start=0):
    """
    Find a lower bound.

    PRECONDITION: The formula should not be UNSAT.

    :return: The binary logarithm of the lower bound
    """
    log.info('find_lower_bound(): phase 1')
    j = 0
    while algorithm1(start + 2**j, 1):
        j += 1
    if j == 0:
        return start
    i = start + 2 ** (j - 1)
    up = start + 2 ** j - 1

    log.info('find_lower_bound(): phase 2  exponent={}'.format(j))
    while i < up:
        m = int(math.ceil((up - i)/2.0) + i)
        if algorithm1(m, 1):
            i = m
        else:
            up = m - 1

    log.info('find_lower_bound(): phase 3  i={}'.format(i))
    if i > 2:
        j = 1
    else:
        j = 0
    while True:
        i -= 2 ** j
        j += 1
        if i <= start or algorithm1(i, conf.alg1_n_iter):
            break
    if i > start:
        ret = i
    else:
        ret = start
    log.info('find_lower_bound(): Returning {} (time elapsed: '
             '{:.2f})'.format(ret, time.process_time()))
    return ret


def find_upper_bound(lower):
    try:
        ub_a, ub_b = algorithm2(lower, conf.alg2_error_prob, conf.alg2_n_iter)
        return ub_a, ub_b
    except SolverBoundsExceededException:
        print('F2: Algorithm 2 Failed!')
        exit(201)


def extract_support_vars():
    """
    Generate the set of vars over which to generate the constraints.
    Extract the independent support or sampling set variables from header lines
    (starting with 'c ind') in the cnf file if they exist.
    Else return the set of all variables of the formula (as implied by the
    'p' header line of the DIMACS cnf file).

    Uses the global var formula_lines which contains the list of lines of
    the cnf file
    :return: A list of variables
    """
    global formula_lines  # The lines of the cnf file
    num_variables = -1
    indset = set()  # The vars of the independent or sampling set, if any
    for l in formula_lines:
        if l.lower().startswith('c ind'):
            for s in l.split(' ')[2:-1]:
                indset.add(int(s))
        if str(l.strip()[0:5]).lower() == 'p cnf':
            fields = l.strip().split(' ')
            num_variables = int(fields[2])
    if num_variables == -1:
        print('F2: Malformed input file. "P CNF" header line is missing!')
        exit(300)
    # if no 'c ind' variables are given, then all variables are presumed to
    # belong to the support
    if len(indset) == 0:
        indset = range(1, num_variables + 1)
    return list(indset), len(indset) > 0


def get_boost(constr):
    """
    Calculate the max boost for a list of constraint numbers
    :param constr: An int or a list of ints representing constraint numbers
    :return: Maximum value of a bound for boost for these constraint numbers
    """
    check_time()
    left_degree = conf.degn
    n = len(conf.var_set)
    if type(constr) == list:
        bs = [boost(left_degree, n, i) for i in constr]
        r = max(bs)
    else:
        assert type(constr) == int
        r = boost(left_degree, n, constr)
    log.info('The Boost is {:.2f}'.format(float(r)))
    return r


def generateConstraints(n_constr, degn, force_long=False, dummy_vars=0):
    if n_constr == 0:
        return list()
    if needLong(n_constr) or force_long:
        return generateLong(n_constr)
    else:
        return generateLDPC(n_constr, dummy_vars, degn)


def needLong(n_constr):
    """
    Are long constraints needed?
    :param n_constr:
    :return:
    """
    assert n_constr > 0
    n = len(conf.var_set)
    degn = conf.degn
    constr_length = math.ceil((n * degn) / n_constr)
    if constr_length >= n / 2:
        return True
    else:
        return False


def need_dummy_var(l, n, i):
    """
    Do we need an extra "dummy" var in the constraint creation
    :param l:         Left (Variable) degree
    :param n:         Num of variables
    :param i:         Num of constraints
    :return: True/False
    """
    return l * n / i == rfloor(l, n, i) and rfloor(l, n, i) % 2 == 0


def generateLong(n_constr):
    """
    Generate random long XOR constraints of average length n/2
    The independent set if any are taken from the
    configuration.
    :param n_constr: The number of constraints to generate
    :return: A list of constraints repr. as lists of strings ending with nl
    """
    assert n_constr > 0
    n = len(conf.var_set)
    var_list = conf.var_set
    i = n_constr
    log.info('Generating Long: n_constr={}  n_var={}'.format(n_constr, n))
    clauses_l = []
    for j in range(int(i)):
        clause_varlist = [v for v in var_list if random.randint(0, 1) > 0]
        is_negated = (random.randint(0, 1) > 0)
        if is_negated:
            out = ['x -']
        else:
            out = ['x ']
        for v in clause_varlist:
            out.append('{} '.format(v))
        out.append('0\n')
        cl = ''.join(out)
        clauses_l.append(cl)
    return clauses_l


def generateLDPC(n_constr, extra_vars, degn):
    """
    Generate the constraints corresponding to an LDPC code.
    The variable set is taken from the configuration. If extra_vars > 0, a
    number of "dummy" vars participate in the generation of the constraints.

    :param n_constr:    The number of constraints to generate
    :param extra_vars:  Number of extra variables
    :param degn:        Average variable (left) degree

    :return: A list of constraints repr. as lists of strings ending with nl
    """
    assert n_constr > 0
    assert extra_vars >= 0
    n0 = len(conf.var_set)
    if extra_vars == 0:
        var_list = conf.var_set
    else:
        var_list = conf.var_set.copy()
        var_list.extend(range(n0 + 1, n0 + extra_vars + 1))
    n = len(var_list)
    i = int(n_constr)
    x = n * degn / i  # Average constraint degree
    log.info('Generating LDPC: n_constr={}  n_var={} var_degree={}  '
             'constr_degree={:.1f}'.format(n_constr, n, degn, x))
    i_degseq = [math.floor(x)] * i
    n_degseq = [math.floor(degn)] * n
    if degn > math.floor(degn):
        x_ = math.ceil((degn - math.floor(degn)) * n)
        indices_ = random.sample(range(n), x_)
        for j_ in indices_:
            n_degseq[j_] += 1
    surplus = sum(n_degseq) - sum(i_degseq)
    assert 0 <= surplus <= n, 'Improper surplus  = ' + str(surplus)
    for c_ in random.sample(range(i), surplus):
        i_degseq[c_] += 1
    g = bipartite.configuration_model(i_degseq, n_degseq,
                                      create_using=nx.Graph())
    clauses_l = []
    for j in range(i):
        clause_varlist = [var_list[ind_ - i] for ind_ in g.neighbors(j)]
        is_negated = (random.randint(0, 1) > 0)
        if is_negated:
            out = ['x -']
        else:
            out = ['x ']
        for v in clause_varlist:
            out.append('{} '.format(v))
        out.append('0\n')
        cl = ''.join(out)
        clauses_l.append(cl)
    return clauses_l


def setup_augmented_formula(n_constr, constraints):
    """
    Create a DIMACS file with the original formula augmented with LDPC XOR
    constraints.
    :param n_constr: The number of XORs to be actually added to the original
                     formula. It may be less or equal to the number of
                     constraints supplied with 'constraints' parameter.
    :param constraints: A list of constraints (repr. as strings ending with
                        newline)
    :return: The full pathname of the augmented formula.
    """
    global aug_counter, formula_lines
    if n_constr == 0:
        return conf.formula_filename
    check_time()
    filename = os.path.splitext(os.path.basename(conf.formula_filename))[0]
    aug_counter += 1
    augmented_filename = '{}_augform_{}.cnf'.format(filename, aug_counter)
    outputfile = os.path.join(conf.working_dir, augmented_filename)
    with open(outputfile, 'w') as ofile:
        for iline in formula_lines:
            if str(iline.strip()[0:5]).lower() == 'p cnf':
                fields = iline.strip().split(' ')
                num_variables = int(fields[2])
                num_clauses = int(fields[3]) + n_constr
                ofile.write('p cnf {} {}\n'.format(num_variables, num_clauses))
                continue
            ofile.write(iline)
        for l in constraints[:int(n_constr)]:
            ofile.write(l)
    return outputfile


def parse_cryptominisat_output(output):
    if output is None:
        return 'ERROR', math.nan, math.nan  # Shouldn't reach anywhere,normally
    version = conf.cmsat_major_version
    ctime = math.nan
    nsol = 0
    res_type = "ERROR"
    if version == 5:
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('c Number of solutions found until now:'):
                nsol = int(line.split('now:')[1].strip())
            elif line.startswith('c Total time'):
                ctime = float(line.split(':')[1].strip())
            elif line.startswith('s SAT'):
                res_type = 'SAT'
                nsol += 1
            elif line.startswith('s UNSAT'):
                res_type = 'UNSAT'
            elif line.startswith('s INDET'):
                res_type = 'INDET'
    elif version == 2:
        res_type = 'INDET'
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('c Number of solutions found until now:'):
                nsol = int(line.split('now:')[1].strip())
            elif line.startswith('c CPU time'):
                ctime = float(line.split(':')[1].strip().split('s')[0].strip())
            elif line.startswith('c SAT'):
                res_type = 'SAT'
                nsol += 1
            elif line.startswith('c UNSAT'):
                res_type = 'UNSAT'
            elif (line.startswith('cryptominisat:') or line ==
                    'Memory manager cannot handle the load. Sorry. Exiting.'):
                log.warning('*** cryptominisat failed!')
                res_type = 'ERROR'
        if nsol > 0:
            res_type = 'SAT'  # Because a 'c UNSAT' is given by CMSAT if maxsol
#                               is not reached.
    else:
        raise Exception('cmsat parsing for this version not yet implemented')
    return res_type, nsol, ctime


def parse_sharpsat_output(output):
    """
    Parse the output of sharpsat and remove the data.out file it
    generates in the current dir.
    If it finds the number of solutions it is reported in the first
    element of the returning pair, otherwise it is math.nan.
    The second element of the returning pair is 0, to conform with the
    protocol for the last element of the returning tuple from output parsers
    :param output: The sharpsat output. If None then a timeout occured
    :return: A pair of values
    """
    if output is None:
        return math.nan, 0
    dataout_file = os.path.join(os.getcwd(), 'data.out')
    if os.path.isfile(dataout_file):
        os.remove(dataout_file)
    ctime = 0
    nsol = math.nan
    all_lines = output.split('\n')
    for i, line in enumerate(all_lines):
        line = line.strip()
        if line.startswith('# solutions'):
            nsol = int(all_lines[i+1].strip())
    return nsol, ctime


def execute_cmd(sh_cmd, timeout, output_parser, plain_timeout=False):
    """ Do the following:
    * Execute the command
    * Save and parse the output
    :param sh_cmd: The cryptominisat command to execute
    :param timeout: A marginaly larger value than the one passed to cmsat
                    directly
    :param output_parser: function to parse the output of the command. The
                          function should return a tuple with last element the
                          processor time that consumed.
    :param plain_timeout: [boolean] If it is True then it is normal the timeout
                          set for the subprocess to be triggered i.e. there
                          is no other 'internal' timeout in the command
                          that should be triggered earlier. Thus if
                          'plain_timeout' is true and a timeout occurs it is
                          not reported as unrecoverable error.
    :return: The parser output
    """
    global conf, subprocess_cumulative_time
    check_time()
    cmd_l = shlex.split(sh_cmd)
    return_code = math.nan
    # noinspection PyUnusedLocal
    res_type = 'ERROR'
    # noinspection PyUnusedLocal
    nsol = math.nan
    ctime = math.nan
    output = None
    log.debug('Executing cmd: "{}"'.format(sh_cmd))
    try:
        cmd_res = subprocess.run(
            cmd_l, cwd=conf.working_dir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout)
        return_code = cmd_res.returncode
        output = cmd_res.stdout.decode('utf-8')

    except subprocess.TimeoutExpired as e:
        if not plain_timeout:
            print('** ', str(e))
            assert False, 'Wrapper timeout triggered before command finishes.'
    finally:
        log.debug('Finished cmd: "{}" with return code:{} time:{}'.format(
                   sh_cmd, return_code, ctime))
        log.debug('TotalTime(s):{}'.format(ctime))
        parsed_results = output_parser(output)
        ctime = parsed_results[-1]
    assert (not math.isnan(ctime))
    subprocess_cumulative_time += ctime
    return parsed_results


def sat_count(formula, time_limit, sol_limit):
    """
    Count the number of solutions of xor+cnf formula.

    :param formula: The full pathname of the formula file
    :param time_limit: Maximum run time (in sec) allowed
    :param sol_limit: Maximum number of solutions to be returned
    :return: a tuple (t, b, n, ctime) such that:
                t: [boolean] has the time_limit been reached?
                b: [boolean] has the max number of solutions be reached?
                n: - min{num_of_solutions, sol_limit}, if 't' is False, or
                   - num of solutions found until time out, if 't' is True
                ctime: The processor time consumed in the subprocess
    """
    check_time()
    cmsat5_cmd = ('{cmsat_exe} --printsol 0  --verb 1'
                  ' --maxtime={timeout} --autodisablegauss 0'
                  ' --maxsol={maxsol} {augm_form}')
    cmsat2_cmd = ('{cmsat_exe} --nosolprint  --verbosity=1'
                  ' --maxtime={timeout} --gaussuntil=400'
                  ' --maxsolutions={maxsol} {augm_form}')
    if conf.cmsat_major_version == 5:
        sh_cmd = cmsat5_cmd.format(timeout=time_limit, augm_form=formula,
                                   cmsat_exe=conf.cmsat_exe, maxsol=sol_limit)
    elif conf.cmsat_major_version == 2:
        sh_cmd = cmsat2_cmd.format(timeout=time_limit, augm_form=formula,
                                   cmsat_exe=conf.cmsat_exe, maxsol=sol_limit)
    else:
        raise Exception('Unknown cmsat version')
    result, n_sol, ctime = execute_cmd(sh_cmd, time_limit + 5,
                                       parse_cryptominisat_output)
    if result == 'ERROR':
        print('sat_count: *** CMSAT ERROR ***')
    return result == 'INDET', n_sol >= sol_limit, n_sol, ctime


def sharpsat_count(formula, time_limit):
    """
    Count the number of solutions of a cnf formula by invoking
    sharpsat.
    If sharpsat returns number of solutions equal to 1, the formula may
    also be UNSATISFIABLE. (bug in sharpsat)
    :param formula: The full pathname of the formula file
    :param time_limit: Maximum run time (in sec) allowed
    :return: a triplet (t, n, ctime) such that:
                t: [boolean] has the time_limit been reached without finishing?
                n: number of solutions (if t is False)
                ctime: The processor time consumed in the subprocess
    """
    global subprocess_cumulative_time
    check_time()
    sh_cmd = '{sharpsat_exe} {form}'.format(form=formula,
                                            sharpsat_exe=conf.sharpsat_exe)
    n_sol, _ = execute_cmd(sh_cmd, time_limit, parse_sharpsat_output,
                           plain_timeout=True)
    # A pessimistic take is that all time is consumed.
    # However if sharpsat terminated earlier then it has successfully
    # counted solutions so we don't care about time any more.
    ctime = time_limit
    subprocess_cumulative_time += ctime
    return n_sol is math.nan, n_sol, ctime


def algorithm1(n_constr, n_iter):
    """
    Determine if 2^(n_constr) is a lower bound with error probability
    exp(-n_iter/8)
    :param n_constr: Number of XOR's to add to the formula
    :param n_iter: Number of iterations ('t' in the paper)
    :return: True or False
    """
    assert(n_constr > 0)
    n_var = len(conf.var_set)
    if n_constr >= n_var:
        return False

    threshold = conf.alg1_threshold_coeff * n_iter
    z = 0  # The capital Z in the paper
    answer = False
    for i in range(1, n_iter+1):
        t1 = time.process_time()
        constraints = generateConstraints(n_constr, conf.degn)
        aug_form = setup_augmented_formula(n_constr, constraints)
        t2 = time.process_time()
        max_time = min(conf.alg1_loop_timeout, remaining_time() + 1)
        t, b, n, ctime = sat_count(aug_form, max_time, conf.alg1_maxsol)
        log.info('Algorithm1:{itr}:{n_constr}:{n_sol}:{setup:.2f}:{run:.2f}:'
                 '{timeout}:{hit_bound}'
                 ''.format(itr=i, n_constr=n_constr, n_sol=n,
                           setup=t2 - t1, run=ctime, timeout=t, hit_bound=b))
        if not conf.keep_cnf:
            os.remove(aug_form)
        z += n
        if z >= threshold:
            answer = True
            break
    return answer


def algorithm2(lb, delta: float=math.nan, num_iter: int=math.nan):
    """
    Calculate an upper bound which is correct with probability 1-delta
    given a correct lower bound.
    :param lb: Floor of the binary log of the lower bound ('l' in the paper)
    :param delta: error probability
    :param num_iter: bypass the rigorous computation and give num of iterations
    :return: (a,b) such that a*2^b is a rigorous upper bound
    """
    assert lb >= 1
    force_long = False
    # Only one of delta or num_iter can be specified
    assert((not math.isnan(delta) and math.isnan(num_iter)) or
           (math.isnan(delta) and not math.isnan(num_iter)))
    if not math.isnan(num_iter):
        n_iter = num_iter
    else:
        bst = 1
        if needLong(lb):
            force_long = True
        else:
            bst = get_boost(lb)  # B in the paper
            if bst == 1:
                force_long = True
        n_iter = int(math.ceil(8*(bst+1)*math.log(1/float(delta))))
    z = 0
    for j in range(1, n_iter+1):
        t1 = time.process_time()
        if not need_dummy_var(conf.degn, len(conf.var_set), lb):
            constraints = generateConstraints(lb, conf.degn, force_long)
        else:
            constraints = generateConstraints(lb, conf.degn, force_long,
                                              dummy_vars=1)
        aug_form = setup_augmented_formula(lb, constraints)
        t2 = time.process_time()
        max_time = min(conf.alg2_loop_timeout, remaining_time())
        t, b, n, ctime = sat_count(aug_form, max_time, conf.alg2_maxsol)
        log.info('Algorithm1:{itr}:{n_constr}:{n_sol}:{setup:.2f}:{run:.2f}:'
                 '{timeout}:{hit_bound}'
                 ''.format(itr=j, n_constr=lb, n_sol=n,
                           setup=t2 - t1, run=ctime, timeout=t, hit_bound=b))
        if not conf.keep_cnf:
            os.remove(aug_form)
        if t or b:
            raise SolverBoundsExceededException(
                'Algorithm2: timeout={}, cutoff={}'.format(t, b))
        z += n
    b = z / float(n_iter)
    if not need_dummy_var(conf.degn, len(conf.var_set), lb):
        e = lb + 1
    else:
        e = lb
    return b, e


def F2_algorithm(lb_in, ub_in, delta_in: float, theta: float):
    """
    Calculates an estimate for the number of solutions with a specified
    confidence and accuracy
    :param lb_in: binary logarithm of the lower bound
    :param ub_in: binary logarithm of the upper bound
    :param delta_in: confidence
    :param theta: probability of error
    :return: a, b such that the estimate is a * 2^b
    """
    check_time()
    log.info('Starting F2_algorithm( lb={:.2f}, ub={:.2f}, delta={}, theta={}'
             ''.format(lb_in, ub_in, delta_in, theta))
    if lb_in < 2 - math.log2(delta_in):  # Small LB phase
        t, b, n, ctime = sat_count(conf.formula_filename, conf.total_max_time,
                                   4 / delta_in)
        if t:
            raise SolverBoundsExceededException(
                'F2_algorithm(Small LB phase): Uncontainable Solver Timeout'
                ' occurred')
    lb = int(math.floor(math.log2(delta_in) + lb_in - 2))
    ub = int(math.ceil(ub_in))
    assert ub - 1 > lb  # The range(lb, ub-1) has at least one element
    bst = get_boost(list(range(lb, ub-1)))

    delta = min(delta_in, 1/3)
    xi = 8 / delta
    b = math.ceil(xi + 2*(xi + xi**2*(bst-1)))
    n_iter = int(math.ceil((2*b**2/9)*math.log(5.0/theta)))

    z_list = [0 for _ in range(lb, ub+1)]

    f2_alg_loop_timeout = remaining_time() / min(5, n_iter*(ub-lb+1))
    force_long = needLong(lb)
    for j in range(1, n_iter+1):
        if not need_dummy_var(conf.degn, len(conf.var_set), lb):
            constraints = generateConstraints(lb, conf.degn, force_long)
        else:
            constraints = generateConstraints(lb, conf.degn, force_long,
                                              dummy_vars=1)
        for i in range(lb, ub+1):
            aug_form = setup_augmented_formula(i, constraints)
            max_time = min(f2_alg_loop_timeout, remaining_time())
            tp, _, y, ctime = sat_count(aug_form, max_time, b)
            if not conf.keep_cnf:
                os.remove(aug_form)
            if tp:
                raise SolverBoundsExceededException(
                    'F2_algorithm(Main phase): Uncontainable Solver Timeout '
                    'occurred')
            z_list[i-lb] += y

    thres = n_iter * (1 - delta) * (4 / delta)
    kl = [i for i in range(lb, ub+1) if z_list[i - lb] > thres]
    if not kl:
        return -1, -1
    else:
        if not need_dummy_var(conf.degn, len(conf.var_set), lb):
            j = kl[-1]
        else:
            j = kl[-1] - 1
        return z_list[j]/n_iter, j


def print_sharpsat_warning():
    msg = """
*** Important Notice: The result is produced by
SharpSAT (https://github.com/marcthurley/sharpSAT/) 
which ignores header lines starting with 'c ind' in the CNF input file.
If the variables specified there represent an INDEPENDENT SUPPORT then you
don't need to do anything. Otherwise ignore the result and run F2 again, 
adding the '--skip-sharpsat' option to the command line.\
"""
    print(msg)


# Boost Calculation related code below
def rfloor(l, n, i):
    return int(sp.floor(l * n / i))


def i0(l, n, i):
    return int(-l * n + i * sp.floor(l * n / i) + i)


def i1(l, n, i):
    return int(l * n - i * sp.floor(l * n / i))


def pl(r):
    return [(sp.binomial(r, 2 * j).evalf(), 2 * j) for j in range(0, r + 1)]


def ql(r):
    return [(sp.binomial(r + 1, 2 * j).evalf(), 2 * j)
            for j in range(0, r + 1)]


def h(t):
    return -t * sp.log(t, 2) - (1 - t) * sp.log(1 - t, 2)


def zp(i, n):
    # Solve h(x)-(i-1)/n == 0 with starting point x_0=0.000000001 where
    # h(x) the binary entropy function: h(x):=-x*log2(x)-(1-x)*log2(1-x)
    t = scipy.optimize.fsolve(
        lambda x: (-x) * math.log2(x) -(1-x) * math.log2(1-x) - (i - 1) / n,
        0.000000001
    )[0] * n
    return int(math.ceil(t))


def power_productl(cel1, ex1, cel2, ex2, order):
    assert order > 0
    assert ex1 > 0
    # Sort the list by increasing exponent
    cel1s = sorted(cel1, key=lambda x: x[1])
    cel2s = sorted(cel2, key=lambda x: x[1])
    # Prune the exponents above 'order' in the initial list
    cel1s = [x for x in cel1s if x[1] < order]
    cel2s = [x for x in cel2s if x[1] < order]
    # Coefficients of powers x^0, x^1, ... x^\(order-1\)
    coef_accumulators = [0] * order
    # Initialize accumulators by first the first factor
    for c, e in cel1s:
        if e < order:
            coef_accumulators[e] += c
    ex1 -= 1
    while ex1 > 0:
        coef_accumulators_snapshot = coef_accumulators.copy()
        for c, e in cel1s:
            for i in range(0, order - e):
                if coef_accumulators_snapshot[i] != 0:
                    if e == 0:
                        coef_accumulators[i] *= c
                    else:
                        coef_accumulators[i + e] += coef_accumulators_snapshot[
                                                        i] * c
        ex1 -= 1
    while ex2 > 0:
        coef_accumulators_snapshot = coef_accumulators.copy()
        for c, e in cel2s:
            for i in range(0, order - e):
                if coef_accumulators_snapshot[i] != 0:
                    if e == 0:
                        coef_accumulators[i] *= c
                    else:
                        coef_accumulators[i + e] += coef_accumulators_snapshot[
                                                        i] * c
        ex2 -= 1
    return coef_accumulators


def codewordsl(l, n, i, dlist):
    ordr = max(dlist) * l + 1
    e1 = pl(rfloor(l, n, i))
    e2 = ql(rfloor(l, n, i))
    expansion = power_productl(e1, i0(l, n, i), e2, i1(l, n, i), ordr)
    cwlist = []
    for d in dlist:
        c = expansion[d * l]
        b = sp.binomial(n, d).evalf() / sp.binomial(n * l, d * l).evalf()
        cwlist.append(b * c)
    return cwlist


def boostl(l, n, i):
    zeta = zp(i, n)
    if zeta < 2:
        return 1
    d_l = list(range(1, zeta))
    s1 = sum(codewordsl(l, n, i, d_l))
    s2 = sum([sp.binomial(n, d2).evalf() for d2 in d_l])
    return s1 / s2


def boost(l, n, i):
    # if the constraint length is >= n/2 then we use long constraints
    if math.ceil((n * l) / i) >= n/2:
        return 1

    if l * n / i == rfloor(l, n, i) and rfloor(l, n, i) % 2 == 0:
        return boostl(l, n + 1, i) * 2 ** i
    else:
        return boostl(l, n, i) * 2 ** i

# #####


if __name__ == '__main__':
    main()
