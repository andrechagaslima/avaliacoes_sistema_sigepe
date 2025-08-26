from src.utils.geral import str2bool, check_if_out_file_exists, check_if_split_exists
from datetime import datetime
import argparse
import random
import socket
import os

def args_llm():
    parser = argparse.ArgumentParser(description='classifies data using llm.')
    
    parser.add_argument('--number_of_examples', type=int, required=True)
    parser.add_argument('--inputdir', type=str, required=True)
    parser.add_argument('--llm_method',    type=str)
    parser.add_argument('--overwrite',     type=lambda x: bool(str2bool(x)), default=False)
    parser.add_argument('--outputdir',     type=str, required=True)
    parser.add_argument('--prompt_dir',    type=str, required=True)
    parser.add_argument('--machine',       type=str, default=socket.gethostname())
    
    args = parser.parse_args()

    args.outfilename    = f"{args.outputdir}/classification.json"
    args.start_cls_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print(args)
    check_if_out_file_exists(args)

    if not os.path.exists(args.outputdir):
        print(f"Criando saida {args.outputdir}")
        os.system("mkdir -p {}".format(args.outputdir))

    random.seed(1608637542)

    info = {
        "args": vars(args),
        "time_to_classify": [],
        "time_to_classify_avg": [],
        "y_pred_text": [],
    }

    return args, info    