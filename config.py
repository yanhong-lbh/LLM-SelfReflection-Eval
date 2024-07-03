import argparse
import hashlib
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('--cache_dir', type=str, default='/share/data/speech/yanhong/KB_retrieval/cache', help='cache directory')
parser.add_argument('--data_dir', type=str, default='/share/data/speech/yanhong/KB_retrieval/data', help='data directory')
parser.add_argument('--dataset', type=str, choices=['truthfulqa', 'hotpotqa'])
parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
parser.add_argument('--api_key', type=str, help='your OpenAI api key')
parser.add_argument('--n_samples', type=int, help='sample n times from LM')

config = parser.parse_args()
