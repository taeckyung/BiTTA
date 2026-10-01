import json
import os
import re
import argparse

import pandas as pd
from multiprocessing import Pool
from functools import partial
import numpy as np 

from pathlib import Path

METHOD_LIST = ["Src", "BN_Stats", "TENT", "EATA", "SAR", "CoTTA", "RoTTA", "SoTTA"]
# METHOD_LIST = ["SimATTA_BIN", "SimATTA"]
# METHOD_LIST = ["BATTA"]

########## cont setting : you must set --cont #################################
# METHOD_LIST = ["TENT", "SAR", "CoTTA", "RoTTA", "SoTTA", "EATA", "BATTA"]     

########## online setting : you must NOT set --cont ###########################
# METHOD_LIST = ["ELPT", "MHPL", "EADA", "CLUE", "DIANA", "BATTA"]              


BASE_DATASET = "imagenetR" # pacs # vlcs # tiny-imagenet

LOG_PREFIX = "eval_results"

SEED_LIST = [0, 1, 2]

DIST = 1

RESET = ""

CORRUPTION_LIST_DICT = {
    
    "pacs" : ["art_painting", "cartoon", "sketch"],
    
    "vlcs" : ["LabelMe", "SUN09", "VOC2007"],
    
    "tiny-imagenet" : ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"],
    "cifar10" : ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"],
    "cifar100" : ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"],
    "imagenet" : ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
                   "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                   "jpeg_compression"],

    "domainnet-126" :  ["clipart", "painting", "sketch"],

    "colored-mnist" :  ["test"],
    
    "imagenetR" :  ["corrupt"],
    
}

def get_avg_online_acc(file_path, random_setting=False):
    f = open(file_path)
    json_data = json.load(f)
    if random_setting:
        pred = np.array(json_data['pred'])
        gt = np.array(json_data['gt'])
        chunk = len(pred) // 4
        res = []
        res += [(pred[ : chunk] == gt[: chunk]).mean() * 100]
        res += [(pred[ : 2 * chunk] == gt[: 2 * chunk]).mean() * 100]
        res += [(pred[ : 3 * chunk] == gt[: 3 * chunk]).mean() * 100]
        res += [(pred == gt).mean() * 100]
        
    else:
        res = json_data['accuracy'][-1]
    
    f.close()
    
    return res 


def process_path(args, path):
    if args.random_setting:
        corruption_list = ['random']
        column_list = ['random1', 'random2', 'random3', 'random4']
    else:
        corruption_list = CORRUPTION_LIST_DICT[args.dataset]
        column_list = corruption_list
    result = {f"{s}": pd.DataFrame(columns=column_list) for s in args.seed}
    method = path.split("/")[-1]
    for (path, _, _) in os.walk(path):
        for corr in corruption_list:
            for seed in args.seed:
                if not args.cont:
                    pattern_of_path = f'.*{corr}.*/'
                else:
                    pattern_of_path = f'.*cont.*/'

                if method in ["Src", "ELPT", "MHPL", "EADA", "CLUE", "DIANA"]:
                    pattern_of_path += f'.*{args.prefix}_{seed}.*'
                else:
                    pattern_of_path += f'{args.prefix}_{seed}_dist{args.dist}.*'

                if args.cont:
                    pattern_of_path += f'{corr}.*'

                pattern_of_path = re.compile(pattern_of_path)
                if pattern_of_path.match(path):
                    if not path.endswith('/cp'):  # ignore cp/ dir
                        try:
                            acc = get_avg_online_acc(os.path.join(path, 'online_eval.json'), random_setting=args.random_setting)
                            if not args.cont:
                                path = '/'.join(path.split('/')[:-3])
                            else:
                                prefix = path.split("/")[-2]
                                path = '/'.join(path.split('/')[:-2])
                            key = method + "_" + f"({path})"
                            if corr == 'random' and args.random_setting:
                                result[f"{seed}"].loc[key, 'random1'] = float(acc[0])
                                result[f"{seed}"].loc[key, 'random2'] = float(acc[1])
                                result[f"{seed}"].loc[key, 'random3'] = float(acc[2])
                                result[f"{seed}"].loc[key, 'random4'] = float(acc[3])

                            else:
                                result[f"{seed}"].loc[key, corr] = float(acc)
                        except Exception as e:
                            pass
    return result


def main(args):
    root = args.root_log + "/" + args.dataset
    paths = [os.path.join(root, f"{method}") for method in args.method]
    with Pool(processes=len(paths)) as p:
        func = partial(process_path, args)
        results = p.map(func, paths)

    for seed in args.seed:
        print(f"SEED:{seed} ")
        result = pd.concat([results[i][f"{seed}"] for i in range(len(results))])
        print(result.to_csv())
        result.to_csv(args.save_path + f"/{seed}_{args.dataset}{'_rand' if args.random_setting else ''}.csv")


def parse_arguments():
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=BASE_DATASET,
                        help='Base dataset')

    parser.add_argument('--method', nargs="*", type=str, default=METHOD_LIST,
                        help='Method name')

    parser.add_argument('--seed', nargs="*", type=int, default=SEED_LIST,
                        help='Seed')

    parser.add_argument('--prefix', type=str, default=LOG_PREFIX,
                        help='Log prefix')

    parser.add_argument('--dist', type=str, default=DIST,
                        help='Distribution')

    parser.add_argument('--cont', default=False, action='store_true',
                        help='Continual learning')

    parser.add_argument('--random_setting', default=False, action='store_true',
                        help='Random Setting')
    
    parser.add_argument('--root_log', type=str, default="log",
                        help='Reset function')
    
    parser.add_argument('--save_path', type=str, default="csvs",
                        help='Direcotry to save csv file')

    parser.add_argument('--suffix', default="", type=str,
                        help='Suffix for folder name')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)


    print(
        f"DATASET: {args.dataset}\n"
        f"LOG_PREFIX: {args.prefix}\n"
        f"METHOD: {args.method}\n"
        f"SEED: {args.seed}\n"
        f"DIST: {args.dist}\n"
        f"CONTINUAL: {args.cont}\n"
    )

    main(args)
