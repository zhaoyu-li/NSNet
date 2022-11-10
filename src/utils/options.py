import argparse


def add_model_options(parser):
    parser.add_argument('--model', type=str, default='NSNet', help='Model choice')

    parser.add_argument('--dim', type=int, default=64, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_rounds', type=int, default=10, help='Number of rounds of message passing')
    parser.add_argument('--n_mlp_layers', type=int, default=3, help='Number of layers in all MLPs')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function in all MLPs')
