import json
import time

from src.llms.llm_for_few_shot import LLM
from src.utils.geral import read_dataset, save_file, get_examples, transform_json

def run_classification(number_of_examples):

    transform_json()

    prompt_dir = './sentiment_analysis/resources/prompt/prompt.json'
    llm_method = 'Llama3.1-I'
    outfilename = './sentiment_analysis/resources/outLLM/sentiment_analysis.json'
    seed = 2024

    info = {}

    df = read_dataset()
    texts_for_few_shot = get_examples(df, prompt_dir, number_of_examples)

    llm = LLM(llm_method=llm_method, prompt_dir=prompt_dir)

    llm.set_model(texts_for_few_shot)

    classification_start_time = time.time()
    print("Predict!")
    y_pred_text = llm.predict(df)
    classification_end_time = time.time()

    info["time_to_classify"] = classification_end_time - classification_start_time
    info["time_to_classify_avg"] = (classification_end_time - classification_start_time) / len(df)
    info["y_pred_text"] = y_pred_text
    info["prompt"] = llm.system_prompt
    info["seed"] = seed

    print(json.dumps(info, indent=4))
    save_file(outfilename, info)

    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

run_classification(number_of_examples=2)