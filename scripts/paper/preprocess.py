import pandas as pd
import os
import json
from datasets import load_dataset


def load_data(input_dir, ref_models):
    """
    Load data from JSON files in the specified input directory for the Wildbench data.

    Args:
        input_dir (str): The directory path where the JSON files are located.
        ref_models (list): A list of reference models.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded Wildbench data.
    """
    data = []
    for ref_model in ref_models:
        path = os.path.join(input_dir, f'ref={ref_model}')
        for file in os.listdir(path):
            if file.endswith('.json'):
                json_out = json.load(open(os.path.join(path, file)))
                json_out = [{'judge': ref_model, **x} for x in json_out]
                data.extend(json_out)
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    # git clone https://github.com/allenai/WildBench.git in main directory
    input_dir = 'WildBench/eval_results/v2.0522/pairwise.v2/eval=gpt-4-turbo-2024-04-09/'
    ref_models = [
        'claude-3-haiku-20240307',
        'gpt-4-turbo-2024-04-09',
        'Llama-2-70b-chat-hf'
    ]

    df = load_data(input_dir, ref_models)
    wb_data = load_dataset("allenai/WildBench", "v2", split="test")

    question_data = wb_data.to_pandas()
    df = df.merge(question_data, on='session_id')
    df.to_json('data/wildbench_pairwise_v2.json', index=False)

    lmsys_data = load_dataset('lmsys/lmsys-arena-human-preference-55k', split='train')
    lmsys_data = lmsys_data.to_pandas()
    lmsys_data.to_csv('data/lmsys.csv', index=False)