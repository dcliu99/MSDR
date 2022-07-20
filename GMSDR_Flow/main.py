from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import torch
import numpy as np

from lib.stsutils import get_adjacency_matrix, load_dataset
from model.pytorch.gmsdr_supervisor import GMSDRSupervisor


TF_ENABLE_ONEDNN_OPTS = 0

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='data/model/PEMS08.yaml', type=str,
                    help='Configuration filename for restoring the model.')
parser.add_argument('--id_filename', default=None, type=str,
                    help='Id filename for dataset')
args = parser.parse_args()


def _init_seed(SEED=10):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
        adj = get_adjacency_matrix(distance_df_filename=config['data']['sensors_distance'],
                                   num_of_vertices=config['model']['num_nodes'],
                                   type_=config['model']['construct_type'],
                                   id_filename=args.id_filename)
        adj_mx = torch.FloatTensor(adj)
        dataloader = load_dataset(dataset_dir=config['data']['data'],
                                  normalizer=config['data']['normalizer'],
                                  batch_size=config['data']['batch_size'],
                                  valid_batch_size=config['data']['batch_size'],
                                  test_batch_size=config['data']['batch_size'],
                                  column_wise=config['data']['column_wise'])
        supervisor = GMSDRSupervisor(adj_mx=adj_mx, dataloader=dataloader, **config)
        supervisor.train()


if __name__ == '__main__':
    _init_seed(10)
    main(args)
