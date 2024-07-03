import os
import json
from tqdm import tqdm
from config import config
from utils import get_response, format_context, load_questions

def get_messages(dataset_name, question, context=None):
    if dataset_name == 'truthfulqa':
        return [{"role": "user", "content": f"{question}"}]
    elif dataset_name == 'hotpotqa' and context:
        formatted_context = format_context(context)
        return [{"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}]
    return []

def save_responses(q_idx, messages, dataset_name, model, data_dir, n_samples):
    init_responses_dir = f'{data_dir}/{dataset_name}_{model}/init_responses_{n_samples}'
    os.makedirs(init_responses_dir, exist_ok=True)
    init_responses_path = f'{init_responses_dir}/{q_idx}.json'

    if not os.path.exists(init_responses_path):
        all_responses = [get_response(messages, model) for _ in range(n_samples)]
        with open(init_responses_path, 'w') as f:
            json.dump(all_responses, f)

def save_critiques(q_idx, question, messages, dataset_name, model, context, data_dir, n_samples):
    init_critiques_dir = f'{data_dir}/{dataset_name}_{model}/init_critiques_{n_samples}'
    os.makedirs(init_critiques_dir, exist_ok=True)
    init_critiques_path = f'{init_critiques_dir}/{q_idx}.json'

    if not os.path.exists(init_critiques_path):
        with open(f'{data_dir}/{dataset_name}_{model}/init_responses_{n_samples}/{q_idx}.json', 'r') as f:
            all_responses = json.load(f)

        all_critiques = []
        for response in all_responses:
            critique_messages = messages + [{"role": "assistant", "content": response}, {"role": "user", "content": "Could you critique your last response?"}]
            result = get_response(critique_messages, model)

            all_critiques.append(result)

        with open(init_critiques_path, 'w') as f:
            json.dump(all_critiques, f)

def assert_responses_and_critiques(q_idx, dataset_name, model, data_dir, n_samples):
    with open(f'{data_dir}/{dataset_name}_{model}/init_responses_{n_samples}/{q_idx}.json', 'r') as f:
        all_responses = json.load(f)
    with open(f'{data_dir}/{dataset_name}_{model}/init_critiques_{n_samples}/{q_idx}.json', 'r') as f:
        all_critiques = json.load(f)

    assert len(all_responses) == n_samples, f"need {n_samples} responses"
    assert len(all_critiques) == n_samples, f"need {n_samples} critiques"

def save_responses_without_reflections(q_idx, question, dataset_name, model, context, data_dir, n_samples):
    res_wo_ref_dir = f'{data_dir}/{dataset_name}_{model}/res_wo_ref_{n_samples}'
    os.makedirs(res_wo_ref_dir, exist_ok=True)
    res_wo_ref_path = f'{res_wo_ref_dir}/{q_idx}.json'

    if not os.path.exists(res_wo_ref_path):
        with open(f'{data_dir}/{dataset_name}_{model}/init_responses_{n_samples}/{q_idx}.json', 'r') as f:
            all_responses = json.load(f)

        messages = get_messages_for_responses(dataset_name, question, all_responses, context)
        res_wo_ref = get_response(messages, model)

        with open(res_wo_ref_path, 'w') as f:
            json.dump(res_wo_ref, f)

def save_responses_with_reflections(q_idx, question, dataset_name, model, context, data_dir, n_samples):
    res_w_ref_dir = f'{data_dir}/{dataset_name}_{model}/res_w_ref_{n_samples}'
    os.makedirs(res_w_ref_dir, exist_ok=True)
    res_w_ref_path = f'{res_w_ref_dir}/{q_idx}.json'

    if not os.path.exists(res_w_ref_path):
        with open(f'{data_dir}/{dataset_name}_{model}/init_responses_{n_samples}/{q_idx}.json', 'r') as f:
            all_responses = json.load(f)
        with open(f'{data_dir}/{dataset_name}_{model}/init_critiques_{n_samples}/{q_idx}.json', 'r') as f:
            all_critiques = json.load(f)

        messages = get_messages_for_responses_with_critiques(dataset_name, question, all_responses, all_critiques, context)
        res_w_ref = get_response(messages, model)

        with open(res_w_ref_path, 'w') as f:
            json.dump(res_w_ref, f)

def get_messages_for_responses(dataset_name, question, all_responses, context):
    messages = []
    if dataset_name == 'truthfulqa':
        for response in all_responses:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": question})
    elif dataset_name == 'hotpotqa':
        formatted_context = format_context(context)
        messages.append({"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."})
        for response in all_responses:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"{question}\n Provide a short answer without explanation."})
    return messages

def get_messages_for_responses_with_critiques(dataset_name, question, all_responses, all_critiques, context):
    messages = []
    if dataset_name == 'truthfulqa':
        for response, critique in zip(all_responses, all_critiques):
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please review and critique your previous response."})
            messages.append({"role": "assistant", "content": critique})
        messages.append({"role": "user", "content": question})
    elif dataset_name == 'hotpotqa':
        formatted_context = format_context(context)
        messages.append({"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."})
        for response, critique in zip(all_responses, all_critiques):
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."})
            messages.append({"role": "assistant", "content": critique})
            messages.append({"role": "user", "content": f"{question}\n Provide a short answer without explanation."})
    return messages

def main():
    cache_dir = config.cache_dir
    data_dir = config.data_dir
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = config.dataset
    model = config.model
    n_samples = config.n_samples

    questions, contexts = load_questions(dataset_name, cache_dir=cache_dir)

    for q_idx, question in tqdm(enumerate(questions), total=len(questions)):
        context = contexts[q_idx] if contexts else None
        messages = get_messages(dataset_name, question, context)

        save_responses(q_idx, messages, dataset_name, model, data_dir, n_samples)
        save_critiques(q_idx, question, messages, dataset_name, model, context, data_dir, n_samples)
        assert_responses_and_critiques(q_idx, dataset_name, model, data_dir, n_samples)
        save_responses_without_reflections(q_idx, question, dataset_name, model, context, data_dir, n_samples)
        save_responses_with_reflections(q_idx, question, dataset_name, model, context, data_dir, n_samples)


if __name__ == "__main__":
    main()

