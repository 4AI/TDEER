[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tdeer-an-efficient-translating-decoding/joint-entity-and-relation-extraction-on-nyt)](https://paperswithcode.com/sota/joint-entity-and-relation-extraction-on-nyt?p=tdeer-an-efficient-translating-decoding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tdeer-an-efficient-translating-decoding/joint-entity-and-relation-extraction-on-1)](https://paperswithcode.com/sota/joint-entity-and-relation-extraction-on-1?p=tdeer-an-efficient-translating-decoding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tdeer-an-efficient-translating-decoding/relation-extraction-on-nyt)](https://paperswithcode.com/sota/relation-extraction-on-nyt?p=tdeer-an-efficient-translating-decoding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tdeer-an-efficient-translating-decoding/relation-extraction-on-webnlg)](https://paperswithcode.com/sota/relation-extraction-on-webnlg?p=tdeer-an-efficient-translating-decoding)


# TDEER 🦌

Official Code For [TDEER: An Efficient Translating Decoding Schema for Joint Extraction of Entities and Relations](https://aclanthology.org/2021.emnlp-main.635/) (EMNLP2021)

## Overview

TDEER is an efficient model for joint extraction of entities and relations. Unlike the common decoding approach that predicts the relation between subject and object, we adopt the proposed translating decoding schema: subject + relation -> objects, to decode triples. By the proposed translating decoding schema, TDEER can handle the overlapping triple problem effectively and efficiently. The following figure is an illustration of our models.

![overview](docs/TDEER-Overview.png)

## Reproduction Steps

### 1. Environment


We conducted experiments under python3.7 and used GPUs device to accelerate computing. 

You should first prepare the tensorflow version in terms of your GPU environment. For tensorflow version, we recommend `tensorflow-gpu==1.15.0`.

Then, you can install the other required dependencies by the following script.

```bash
pip install -r requirements.txt
```


### 2. Prepare Data

We follow [weizhepei/CasRel](https://github.com/weizhepei/CasRel) to prepare datas.

For convenience, we have uploaded our processed data in this repository via git-lfs. To use the processed data, you could download the data and decompress it (`data.zip`) into the `data` folder.


### 3. Download Pretrained BERT


Click 👉[BERT-Base-Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) to download the pretrained model and then decompress to `pretrained-bert` folder.


### 4. Train & Eval

You can use `run.py` with `--do_train` to train the model. After training, you can also use `run.py` with `--do_test` to evaluate data.

Our training and evaluating commands are as follows:

1\. NYT

train:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u run.py \
--do_train \
--model_name NYT \
--rel_path data/NYT/rel2id.json \
--train_path data/NYT/train_triples.json \
--dev_path data/NYT/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--save_path ckpts/nyt.model \
--learning_rate 0.00005 \
--neg_samples 2 \
--epoch 200 \
--verbose 2 > nyt.log &
```

evaluate:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
--do_test \
--model_name NYT \
--rel_path data/NYT/rel2id.json \
--test_path data/NYT/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--ckpt_path ckpts/nyt.model \
--max_len 512 \
--verbose 1
```

You can evaluate other data by specifying `--test_path`.

2\. WebNLG

train:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u run.py \
--do_train \
--model_name WebNLG \
--rel_path data/WebNLG/rel2id.json \
--train_path data/WebNLG/train_triples.json \
--dev_path data/WebNLG/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--save_path ckpts/webnlg.model \
--max_sample_triples 5 \
--neg_samples 5 \
--learning_rate 0.00005 \
--epoch 300 \
--verbose 2 > webnlg.log &
```

evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
--do_test \
--model_name WebNLG \
--rel_path data/WebNLG/rel2id.json \
--test_path data/WebNLG/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--ckpt_path ckpts/webnlg.model \
--max_len 512 \
--verbose 1
```

You can evaluate other data by specifying `--test_path`.


3\. NYT11-HRL

train:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u run.py \
--do_train \
--model_name NYT11-HRL \
--rel_path data/NYT11-HRL/rel2id.json \
--train_path data/NYT11-HRL/train_triples.json \
--dev_path data/NYT11-HRL/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--save_path ckpts/nyt11hrl.model \
--learning_rate 0.00005 \
--neg_samples 1 \
--epoch 100 \
--verbose 2 > nyt11hrl.log &
```

evaluate:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
--do_test \
--model_name NYT11-HRL \
--rel_path data/NYT/rel2id.json \
--test_path data/NYT11-HRL/test_triples.json \
--bert_dir pretrained-bert/cased_L-12_H-768_A-12 \
--ckpt_path ckpts/nyt11hrl.model \
--max_len 512 \
--verbose 1
```


### Pre-trained Models

We released our pre-trained models for NYT, WebNLG, and NYT11-HRL datasets, and uploaded them to this repository via git-lfs.

You can download pre-trained models and then decompress them (`ckpts.zip`) to the `ckpts` folder.

To use the pre-trained models, you need to download our processed datasets and specify `--rel_path` to our processed `rel2id.json`.

To evaluate by the pre-trained models, you can use above commands and specify `--ckpt_path` to specific model.


In our setting, NYT, WebNLG, and NYT11-HRL achieve the best result on Epoch 86, 174, and 23 respectively.

1\. NYT

<details>
<summary>click to show the result screenshot.</summary>

![](docs/nyt_train_screenshot.png)

</details>

2\. WebNLG

<details>
<summary>click to show the result screenshot.</summary>

![](docs/webnlg_train_screenshot.png)

</details>


3\. NYT11-HRL

<details>
<summary>click to show the result screenshot.</summary>

![](docs/nyt11hrl_train_screenshot.png)

</details>

## Citation

If you use our code in your research, please cite our work:


```bibtex
@inproceedings{li-etal-2021-tdeer,
    title = "{TDEER}: An Efficient Translating Decoding Schema for Joint Extraction of Entities and Relations",
    author = "Li, Xianming  and
      Luo, Xiaotian  and
      Dong, Chenghao  and
      Yang, Daichuan  and
      Luan, Beidi  and
      He, Zhen",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.635",
    pages = "8055--8064",
}

```

## Acknowledgement

Some of our codes are inspired by [weizhepei/CasRel](https://github.com/weizhepei/CasRel). Thanks for their excellent work.


## Contact

If you have any questions about the paper or code, you can

1) create an issue in this repo;
2) feel free to contact 1st author at niming.lxm@alipay.com / xmlee97@gmail.com, I will reply ASAP.
