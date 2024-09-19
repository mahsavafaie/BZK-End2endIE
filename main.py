import argparse
import logging
from inferable.data import get_datasets
from inferable.models import get_models
from inferable.evaluation.evaluator import evaluate
from typing import List
import os
import sys
import random

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--models", "-m", nargs='+', help="<Required> Names of models e.g. `SpacyEntities`", required=True
    )

    parser.add_argument(
        "--datasets", "-d", nargs='+', help="Name of datasets e.g. `inspec`", default=["bzk"]
    )

    parser.add_argument(
        "--output", "-o", type=dir_path, help="Folder of the output files", default="output"
    )
    
    parser.add_argument(
        "--gpu", "-g", help="The GPUs to use (will be passed to CUDA_VISIBLE_DEVICES) e.g. `0,1` or '0'"
    )

    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

    return parser

def parse_eval_args(parser: argparse.ArgumentParser, cmd_arguments: List[str]) -> argparse.Namespace:
    args = parser.parse_args(args=cmd_arguments)

    log_format = "%(asctime)s %(levelname)s %(message)s"

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting random seeds
    seed = 42
    random.seed(seed)
    import numpy as np 
    np.random.seed(seed) # importing here because of cuda visible devices set before
    import torch
    torch.manual_seed(seed)

    return args


def cli_evaluate() -> None:
    cmd_arguments = sys.argv[1:]
    #cmd_arguments = [
        #"-d", "bzk_small", "-m", "class=PaliGemma", "-g", "1"
        #"-d", "bzk_small", "-m", "class=Dummy"
        #"-d", "bzk_small", "-m", "class=Donut,model_name=naver-clova-ix/donut-base-finetuned-cord-v2"
        #"-d", "bzk_small", "-m", "class=Donut"
        #"-d", "bzk_small", "-m", "class=PaliGemmaZeroShot"
        #"-d", "bzk_small", "-m", "class=DonutModelZeroShot,model_name=output/donut_model/2024-07-27_08-40-01,task_prompt=<s_wieder>" 
    #]


    parser = setup_parser()
    args = parse_eval_args(parser, cmd_arguments)
    models = get_models(args.models)
    datasets = get_datasets(args.datasets)
    
    evaluate(models, datasets, args.output)

if __name__ == "__main__":
    cli_evaluate()