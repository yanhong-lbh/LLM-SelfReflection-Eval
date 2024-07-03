# LLM-SelfReflection-Eval 

This code is associated with the NAACL 2024 Findings paper: **"When Hindsight is Not 20/20: Testing Limits on Reflective Thinking in Large Language Models"**. If you use this code or the results from our paper, please cite:

```
@inproceedings{li-etal-2024-hindsight,
    title = "When Hindsight is Not 20/20: Testing Limits on Reflective Thinking in Large Language Models",
    author = "Li, Yanhong  and Yang, Chenghao  and Ettinger, Allyson",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.237"
}
```

## Setup Environment

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Generating Data

### Initial Responses and Critiques

To generate initial response and critique pairs, as well as responses without and with reflection, use the following command. We used `n_samples = 4` in our paper:

```
python generate.py \
    --cache_dir YOUR_CACHE_DIR \
    --data_dir YOUR_DATA_DIR \
    --dataset truthfulqa \
    --api_key YOUR_API_KEY \
    --n_samples 4 \
    --model gpt-3.5-turbo
```

### Synthetic Datastore for HotpotQA

To build a synthetic datastore for HotpotQA, run:

```
python build_synthetic_dataset.py \
    --cache_dir YOUR_CACHE_DIR \
    --data_dir YOUR_DATA_DIR \
    --api_key YOUR_API_KEY \
    --model gpt-3.5-turbo
```

### Generate with Synthetic Data

To generate data using the synthetic datastore, use the following command:

```
python generate_w_synthetic_data.py \
    --cache_dir YOUR_CACHE_DIR \
    --data_dir YOUR_DATA_DIR \
    --dataset hotpotqa \
    --api_key YOUR_API_KEY \
    --n_samples 10 \
    --model gpt-3.5-turbo
```

## Note

The data used for plotting the figures in the paper are in the `plots` directory. You can run `python plots/plot_4.py` and `python plots/plot_10.py` to reproduce Figures 2 and 3 in our paper. Our generated results are saved in the `results` folder for your reference.

