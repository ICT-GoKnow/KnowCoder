import argparse
import os
from tqdm import tqdm

from utils import dump_json_file, read_json_file



def combine(args):
    all_metric_res = {}
    for task in args.tasks.split(','):
        if task not in all_metric_res:
            all_metric_res[task] = {}
        for ckpt in args.ckpts.split(','):
            if ckpt not in all_metric_res[task]:
                all_metric_res[task][ckpt] = {}
            for match_type in args.match_types.split(','):
                res_file = f'{args.data_dir}/{task}/{ckpt}/result_overall_{match_type}.json'
                res = read_json_file(res_file)
                for metric_name, metric_data in res.items():
                    all_metric_res[task][ckpt][metric_name] = metric_data

    dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_result.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='EAE,NER,RE', help='involved tasks')
    parser.add_argument('--match_types', type=str, default='HM,EM', help='match types')
    parser.add_argument('--ckpts', type=str, default='200,400,600,800,1000,1200,1400', help='checkpoints')
    parser.add_argument('--data_dir', type=str, default='intermediate_data', help='intermediate data dir')
    parser.add_argument('--output_dir', type=str, help='the path of output directory.')
    args = parser.parse_args()
    combine(args)