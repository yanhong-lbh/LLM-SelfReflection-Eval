
import os
import csv
import json
import ast
from config import config
from utils import get_response, format_context, load_questions 
from tqdm import tqdm
import random



def load_data(filepath, question_key, context_prefix, num_contexts):
    questions = []
    contexts = []
    with open(filepath, "r") as f:
        data = csv.DictReader(f)
        for row in data:
            curr_questions = [row[question_key] for _ in range(num_contexts)]
            curr_contexts = [ast.literal_eval(row[f'{context_prefix}_{i}']) for i in range(num_contexts)]
            questions.append(curr_questions)
            contexts.append(curr_contexts)
    return questions, contexts

def load_gt_data(filepath, response_prefix, critique_prefix, num_contexts):
    gt_responses = []
    gt_critiques = []
    with open(filepath, "r") as f:
        data = csv.DictReader(f)
        for row in data:
            curr_responses = [row[f'{response_prefix}_{i+1}'] for i in range(num_contexts)]
            curr_critiques = [row[f'{critique_prefix}_{i+1}'] for i in range(num_contexts)]
            gt_responses.append(curr_responses)
            gt_critiques.append(curr_critiques)
    return gt_responses, gt_critiques

def generate_response_path(base_dir, dataset, file_type, q_idx):
    dir_path = os.path.join(base_dir, dataset, file_type)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f'{q_idx}.json')

def generate_messages(question, context, responses=None, critiques=None):
    formatted_context = format_context(context)
    messages = [{"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}]
    if responses:
        for response in responses:
            messages.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"{question}\n Provide a short answer without explanation."}
            ])
    if critiques:
        for response, critique in zip(responses, critiques):
            messages.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
                {"role": "assistant", "content": critique},
                {"role": "user", "content": f"{question}\n Provide a short answer without explanation."}
            ])
    return messages

def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def process_responses_and_critiques(q_idx, questions, contexts, dataset, base_dir, model, tokenizer):
    responses_path = generate_response_path(base_dir, dataset, 'init_responses_10', q_idx)
    critiques_path = generate_response_path(base_dir, dataset, 'init_critiques_10', q_idx)
    
    if not os.path.exists(responses_path):
        all_responses = []
        for i in range(10):
            messages = generate_messages(questions[i], contexts[i])
            result = get_response(messages, model, tokenizer)
            all_responses.append(result)
        save_data(responses_path, all_responses)

    if not os.path.exists(critiques_path):
        with open(responses_path, 'r') as f:
            all_responses = json.load(f)
        
        all_critiques = []
        for i in range(10):
            messages = generate_messages(questions[i], contexts[i], responses=[all_responses[i]])
            messages.append({"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."})
            result = get_response(messages, model, tokenizer)
            all_critiques.append(result)
        save_data(critiques_path, all_critiques)

def get_messages_with_context(question, formatted_context, new_res):
    messages = [{"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."}]
    for i in range(10):
        messages.extend([
            {"role": "assistant", "content": f"{new_res[i]}"},
            {"role": "user", "content": f"{question}\n Provide a short answer without explanation."}
        ])
    return messages

def get_messages_with_context_and_critiques(question, formatted_context, new_res, new_c):
    messages = []
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Context: {formatted_context}\n Question: {question} \n Provide a short answer without explanation."},
            {"role": "assistant","content": f"{new_res[i]}"},
            {"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."},
            {"role": "assistant","content": f"{new_c[i]}"},
        ])
    messages.append({"role": "user", "content": f"{question}\n Provide a short answer without explanation."})
    return messages


def process_responses(q_idx, questions, contexts, dataset, base_dir, model):
    responses_path = generate_response_path(base_dir, f'{dataset}_{model}', 'init_responses_10', q_idx)
    critiques_path = generate_response_path(base_dir, f'{dataset}_{model}', 'init_critiques_10', q_idx)
    
    if not os.path.exists(responses_path):
        all_responses = []
        for i in range(10):
            messages = generate_messages(questions[i], contexts[i])
            result = get_response(messages, model)
            all_responses.append(result)
        save_data(responses_path, all_responses)

    if not os.path.exists(critiques_path):
        with open(responses_path, 'r') as f:
            all_responses = json.load(f)
        
        all_critiques = []
        for i in range(10):
            messages = generate_messages(questions[i], contexts[i], responses=[all_responses[i]])
            messages.append({"role": "user", "content": "Please review and critique your previous response. You can refer back to the original context if needed."})
            result = get_response(messages, model)
            all_critiques.append(result)
        save_data(critiques_path, all_critiques)


def process_responses_w_wo_reflections(q_idx, question_list, context, dataset, data_dir, model, all_gt_res, all_gt_c):

    init_responses_path = generate_response_path(data_dir, f'{dataset}_{model}', 'init_responses_10', q_idx)
    init_critiques_path = generate_response_path(data_dir, f'{dataset}_{model}', 'init_critiques_10', q_idx)

    random.seed(42)
    formatted_context = format_context(context)
    
    res_wo_ref_dir = os.path.join(data_dir, f'{dataset}_{model}', 'res_wo_ref_fake')
    res_w_ref_dir = os.path.join(data_dir, f'{dataset}_{model}', 'res_w_ref_fake')
    os.makedirs(res_wo_ref_dir, exist_ok=True)
    os.makedirs(res_w_ref_dir, exist_ok=True)
    
    res_wo_ref_path = os.path.join(res_wo_ref_dir, f'{q_idx}.json')
    res_w_ref_path = os.path.join(res_w_ref_dir, f'{q_idx}.json')

    if not os.path.exists(res_wo_ref_path):
        with open(init_responses_path, 'r') as f:
            all_responses = json.load(f)
        wo_ref_all_cm = []
        for i in range(11):
            selected_gt_indices = random.sample(range(10), i)
            new_res = [all_gt_res[q_idx][idx] if idx in selected_gt_indices else all_responses[idx] for idx in range(10)]
            messages = get_messages_with_context(question_list[0], formatted_context, new_res)
            res_wo_ref = get_response(messages, model)
            wo_ref_all_cm.append(res_wo_ref)
        with open(res_wo_ref_path, 'w') as f:
            json.dump(wo_ref_all_cm, f)

    if init_critiques_path and not os.path.exists(res_w_ref_path):
        with open(init_responses_path, 'r') as f:
            all_responses = json.load(f)
        with open(init_critiques_path, 'r') as f:
            all_critiques = json.load(f)
        w_ref_all_cm = []
        for i in range(11):
            selected_gt_indices = random.sample(range(10), i)
            new_res = [all_gt_res[q_idx][idx] if idx in selected_gt_indices else all_responses[idx] for idx in range(10)]
            new_c = [all_gt_c[q_idx][idx] if idx in selected_gt_indices else all_critiques[idx] for idx in range(10)]
            messages = get_messages_with_context_and_critiques(question_list[0], formatted_context, new_res, new_c)
            res_w_ref = get_response(messages, model)
            w_ref_all_cm.append(res_w_ref)
        with open(res_w_ref_path, 'w') as f:
            json.dump(w_ref_all_cm, f)


def main():
    cache_dir = config.cache_dir
    data_dir = config.data_dir
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    dataset = config.dataset
    model = config.model
    n_samples = config.n_samples

    all_questions, all_contexts = load_data("results/questions_and_fake_sf.csv", 'question', 'fake_sf', 10)
    all_gt_res, all_gt_c = load_gt_data("results/questions_and_fake_sf_ground_truth.csv", 'response', 'critique', 10)

    original_questions, original_contexts = load_questions(dataset, cache_dir=cache_dir)

    flattened_contexts = [original_contexts[original_questions.index(q[0])] for q in all_questions]

    for q_idx, question_list in tqdm(enumerate(all_questions), total=len(all_questions)):

        process_responses(
            q_idx, question_list, all_contexts[q_idx], dataset, data_dir, model 
        )

        context = flattened_contexts[q_idx]
        process_responses_w_wo_reflections(
            q_idx, question_list, context, dataset, data_dir, model, 
            all_gt_res, all_gt_c
        )

if __name__ == "__main__":
    main()
