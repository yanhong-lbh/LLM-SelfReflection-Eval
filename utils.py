from config import config
from datasets import load_dataset
from openai import OpenAI

def format_context(context):
    formatted_context = ""
    for title, sentences in zip(context['title'], context['sentences']):
        formatted_context += f"{title}: " + " ".join(sentences) + " "
    return formatted_context

def load_questions(dataset_name, cache_dir=None):
    if dataset_name == 'truthfulqa':
        raw_datasets = load_dataset('domenicrosati/TruthfulQA', cache_dir=cache_dir)
        all_questions = raw_datasets['train']['Question']
    elif dataset_name == 'hotpotqa':
        raw_datasets = load_dataset('hotpot_qa', 'distractor', trust_remote_code=True, cache_dir=cache_dir)['train'][:10000]
        all_questions = raw_datasets['question']
        all_contexts = raw_datasets['context']
    return all_questions, all_contexts if dataset_name == 'hotpotqa' else None


def get_response(messages, model):
    client = OpenAI(api_key=config.api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
        n=1,
    )

    return response.choices[0].message.content

