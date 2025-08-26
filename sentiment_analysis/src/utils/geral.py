import os
import json
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def transform_json():

    input_path = "./txt_data/sentiment_analysis.json"
    output_path = "./sentiment_analysis/resources/prompt/prompt.json"

    with open(input_path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    output_json = {
        "system_prompt": (
            "Classify the following texts, which are comments in Portuguese about a government-developed application, "
            "into one of the following categories: criticism, suggestion, positive feedback, or not pertinent. "
            "Classify as 'not pertinent' only texts that are neither suggestions, positive feedback, nor criticisms, "
            "considering that they don't fit those categories but aren't necessarily irrelevant. These comments were provided "
            "by users who were encouraged to give suggestions, critiques, or positive feedback. The response must consist solely "
            "of the name of one of these categories, with no additional text or information."
        ),
        "examples": input_json,
        "categories": ["criticism", "suggestion", "positive feedback", "not pertinent"]
    }


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)


def str2bool(x):
    if x.lower() in ['y', 'yes', 's', 'sim', '1', 'abacaxi']:
        return True
    return False

def check_if_out_file_exists(args):
    if os.path.exists(args.outfilename):
        print("Out dir already exist!")
        exit()

def check_if_split_exists(args):

    # if args.sel == "":
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_idxinfold.csv"
    # else:
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_"+args.sel+"_idxinfold.csv"

    saida = args.filename+".json"

    if os.path.exists(saida):
        print("Already exists selection output")
        exit()

def read_dataset():
    df = pd.read_csv("./data/dataFrame.csv")
    df = df[['ID', 'comments']]
    df.dropna(subset=['comments'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def save_file(save_dir, info):
    with open(save_dir, 'w') as arquivo_json:
        json.dump(info, arquivo_json, indent=4)

def print_in_file(msg, filename):
    with open(filename, 'a') as arq:
        arq.write(msg+"\n")


def get_examples(df, prompt_dir, number_of_examples):
    with open(prompt_dir, 'r') as f:
        data = json.load(f)
    examples = data["examples"]  

    id_col = "ID" if "ID" in df.columns else "id"

    df = df[[id_col, "comments"]].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df.dropna(subset=[id_col])
    df[id_col] = df[id_col].astype(int)

    id_to_comment = df.set_index(id_col)["comments"].to_dict()

    labels_text_lengths = {}
    missing = []
    for idx_str, label in examples.items():
        idx = int(idx_str)
        text = id_to_comment.get(idx)
        if text is None:
            missing.append(idx)
            continue
        labels_text_lengths.setdefault(label, {})[idx] = len(text)

    if missing:
        print(f"Atenção: {len(missing)} ID(s) do JSON não encontrados no dataset: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")

    texts_for_few_shot = {}
    for label, lens in labels_text_lengths.items():
        top_ids = [i for i, _ in sorted(lens.items(), key=lambda x: x[1], reverse=True)[:number_of_examples]]
        for i in top_ids:
            texts_for_few_shot[i] = {"text": id_to_comment[i], "label": label}

    return texts_for_few_shot