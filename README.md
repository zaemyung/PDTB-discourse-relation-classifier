# PDTB-discourse-relation-classifier
- A simple PDTB discourse relation classifier built on top of the RoBERTa model.
- It classifies a pair of two sentences or phrases into one of the 11 classes as defined below:
  ![class_mapping](https://user-images.githubusercontent.com/3746478/143455314-71f783d1-171c-4456-878d-6a93c6e666d8.png)
- The model can be trained from the preprocessed dataset in `data`.

## Requirements
- Python 3.6+
- Pytorch
- Huggingface
- Pytorch Lightning
- TorchMetrics

## Training
```bash
python discourse_baseline.py
```

## Inference
- Prepare two sentences (or phrases) that a discourse relation is to be classified, and save them to `first_sent_path` and `second_sent_path` respectively.
- Run `python prepare_silver_dt.py n_parts split_ind first_sent_path second_sent_path`
  - `n_parts`: the number of data splits for the data.
  - `split_id`: current index of the data splits
- So if you want to split the corpus into three parts, and conduct inference on the first part, it would be:
  - `python prepare_silver_dt.py 3 0 path/to/first_sent path/to/second_sent`
  - The second and third parts can be run separately on different SLURM servers
