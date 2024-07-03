
import pandas as pd
import re
from datasets import load_dataset
import numpy as np
import copy
import os
import csv
import ast
from tqdm import tqdm
from utils import get_response
from config import config


def get_api_response(question, real_sf, model):

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Here is a question: {question}. Please create 10 different versions of 'fake supporting facts' based on the following real supporting facts. Modify only one sentence in each version, making sure the modified sentence is still relevant but contains false information. Keep the other sentences unmodified. Each version of fake supporting facts should have the same number of sentences as the real supporting facts."},
        {"role": "user", "content": f"Real Supporting Facts: {real_sf}"},
        {"role": "user", "content": "Please generate the fake supporting facts versions."},
        {"role": "user", "content": f"Fake Supporting Facts Version 1:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 2:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 3:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 4:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 5:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 6:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 7:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 8:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 9:\n[Insert manipulated sentences here]\nFake Supporting Facts Version 10:\n[Insert manipulated sentences here]"}
    ]

    return get_response(messages, model)

def format_sentences(sentences):
    sentences = ast.literal_eval(sentences)
    return "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])

def process_csv(input_file):
    data = pd.read_csv(input_file)
    data["real_sf"] = data["extracted_sentences"].apply(format_sentences)
    
    if not os.path.exists(f"{config.data_dir}/fake_evidence_unprocessed.csv"):
        questions, real_sfs = data['question'].tolist(), data['real_sf'].tolist()
        processed_data = []

        for question, real_sf in tqdm(zip(questions, real_sfs), total=len(questions)):
            response = get_api_response(question, real_sf, config.model)
            processed_data.append({"question": question, "fake_sf": response})

        # Save to file
        with open(f"{config.data_dir}/fake_evidence_unprocessed.csv", "w", newline='') as csvfile:
            fieldnames = ["question", "fake_sf"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_data)

def split_fake_sf(row):
    versions = [v.strip() for v in re.split(r'Fake Supporting Facts Version \d{1,2}:', row) if v.strip()]
    return [re.split(r'\d{1,2}\.\s', version) for version in versions]

def create_fake_sf_columns(df):
    fake_sf_columns = df["fake_sf"].apply(split_fake_sf)
    for i in range(10):
        df[f"fake_sf_{i}"] = fake_sf_columns.apply(lambda x: x[i] if len(x) > i else None)
    return df

def match_and_modify_context(df, hotpot_qa):
    output_data = []

    for i, row in df.iterrows():
        input_question = row["question"]
        matched_question = next((item for item in hotpot_qa if item['question'] == input_question), None)

        if matched_question:
            contexts = [copy.deepcopy(matched_question['context']) for _ in range(10)]
            supporting_facts = matched_question['supporting_facts']

            for j, fake_sf in enumerate([row[f"fake_sf_{i}"] for i in range(10)]):
                for title, sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
                    if title in contexts[j]['title']:
                        title_idx = contexts[j]['title'].index(title)
                        contexts[j]['sentences'][title_idx][sent_id] = fake_sf[j] if fake_sf and j < len(fake_sf) else ""

            output_data.append({
                'question': input_question,
                **{f"fake_sf_{i}": contexts[i] for i in range(10)}
            })

    return pd.DataFrame(output_data)


def main():

    data_dir = config.data_dir

    input_file = f"results/questions_for_synthetic_dataset.csv"
    process_csv(input_file)

    df = pd.read_csv(f"{data_dir}/fake_evidence_unprocessed.csv")
    df = create_fake_sf_columns(df)

    hotpot_qa = load_dataset("hotpot_qa", "distractor", split="train", cache_dir=config.cache_dir)
    output_df = match_and_modify_context(df, hotpot_qa)

    output_df.to_csv(f"{data_dir}/questions_and_fake_sf.csv", index=False)



if __name__ == "__main__":
    main()
