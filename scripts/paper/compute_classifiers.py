from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shared_biases_human import load_lmsys_local
import torch
from tqdm import tqdm
import pandas as pd


def compute_classifiers(data, model, tokenizer, batch_size=4, index=1):
    """
    Computes the formalities and sentiment of a given dataset using a specified model and tokenizer.

    Args:
        data (list): The input dataset.
        model: The model used for computing formalities and sentiment.
        tokenizer: The tokenizer used for encoding the input data.
        batch_size (int, optional): The batch size for processing the data. Defaults to 4.
        index (int, optional): The index of the formality or sentiment to extract from the model's output. Defaults to 1.

    Returns:
        list: A list of formalities or sentiments computed for each input in the dataset.
    """
    formalities = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        # batch = ['\n'.join(x) for x in batch]
        tokenized_batch = tokenizer.batch_encode_plus(batch, return_tensors='pt', 
                                                      padding=True).to(model.device)
        formality = torch.softmax(model(input_ids=tokenized_batch['input_ids'][:, :512], 
                                        attention_mask=tokenized_batch['attention_mask'][:, :512]).logits, dim=-1)
        formality = list(formality.detach().cpu().numpy()[:, index])
        formalities.extend(formality)
    return formalities

if __name__ == '__main__':
    wildbench_data = pd.read_json('data/wildbench_pairwise_v2.json')

    all_a_samples = [row['model_outputs'][row['model_A']] for i, row in wildbench_data.iterrows()]
    all_b_samples = [row['model_outputs'][row['model_B']] for i, row in wildbench_data.iterrows()]

    if 'formality_a' not in wildbench_data.columns:
        model = 's-nlp/roberta-base-formality-ranker'

        formality_model = AutoModelForSequenceClassification.from_pretrained(model)
        if torch.cuda.is_available():
            formality_model = formality_model.to('cuda:0')

        tokenizer = AutoTokenizer.from_pretrained(model)
        wildbench_data['formality_a'] = compute_classifiers(all_a_samples, formality_model, tokenizer)
        wildbench_data['formality_b'] = compute_classifiers(all_b_samples, formality_model, tokenizer)

    if 'sentiment_a' not in wildbench_data.columns:
        sentiment = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment)
        if torch.cuda.is_available():
            sentiment_model = sentiment_model.to('cuda:0')
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment)
        wildbench_data['sentiment_a'] = compute_classifiers(all_a_samples, sentiment_model, sentiment_tokenizer, index=2)
        wildbench_data['sentiment_b'] = compute_classifiers(all_b_samples, sentiment_model, sentiment_tokenizer, index=2)

    wildbench_data.to_json('data/wildbench_pairwise_v2.json', index=False)

    lmsys_data = load_lmsys_local()
    all_a_samples = lmsys_data['response_a'].to_list()
    all_a_samples = ['\n'.join(x) for x in all_a_samples]
    all_b_samples = lmsys_data['response_b'].to_list()
    all_b_samples = ['\n'.join(x) for x in all_b_samples]

    if 'formality_a' not in lmsys_data.columns:
        model = 's-nlp/roberta-base-formality-ranker'

        formality_model = AutoModelForSequenceClassification.from_pretrained(model)
        if torch.cuda.is_available():
            formality_model = formality_model.to('cuda:0')

        tokenizer = AutoTokenizer.from_pretrained(model)
        lmsys_data['formality_a'] = compute_classifiers(all_a_samples, formality_model, tokenizer)
        lmsys_data['formality_b'] = compute_classifiers(all_b_samples, formality_model, tokenizer)

    if 'sentiment_a' not in lmsys_data.columns:
        sentiment = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment)
        if torch.cuda.is_available():
            sentiment_model = sentiment_model.to('cuda:0')
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment)
        lmsys_data['sentiment_a'] = compute_classifiers(all_a_samples, sentiment_model, sentiment_tokenizer, index=2)
        lmsys_data['sentiment_b'] = compute_classifiers(all_b_samples, sentiment_model, sentiment_tokenizer, index=2)

    lmsys_data.to_csv('data/lmsys.csv', index=False)
