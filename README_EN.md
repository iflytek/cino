[**中文说明**](https://github.com/ymcui/Chinese-Minority-PLM/) | [**English**](https://github.com/ymcui/Chinese-Minority-PLM/blob/main/README_EN.md)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="500"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/Chinese-Minority-PLM/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-Minority-PLM.svg?color=blue&style=flat-square">
    </a>
</p>
Pre-trained Language Model (PLM) has been an important technique in the recent natural language processing field, including multilingual NLP research. In order to promote the NLP research for Chinese minority languages, the Joint Laboratory of HIT and iFLYTEK Research (HFL) has released the first specialized pre-trained language model **CINO** (**C**hinese m**INO**rity PLM).

You may also be interested in,

- Chinese MacBERT: https://github.com/ymcui/MacBERT
- Chinese ELECTRA: https://github.com/ymcui/Chinese-ELECTRA
- Chinese BERT series: https://github.com/ymcui/Chinese-BERT-wwm
- Chinese XLNet: https://github.com/ymcui/Chinese-XLNet
- Knowledge Distillation Toolkit - TextBrewer: https://github.com/airaria/TextBrewer
- Model Pruning Toolkit - TextPruner：https://github.com/airaria/TextPruner

More resources by HFL: https://github.com/ymcui/HFL-Anthology

## News

Jan 25, 2022 **CINO-base-v2, CINO-large-v2, and WCM-v2 have been released**.

Dec 17, 2021 **We have released a model pruning toolkit TextPruner**. Check https://github.com/airaria/TextPruner

Oct 25, 2021 CINO-large and Wiki-Chinese-Minority（WCM）dataset have been released.

## Guide
| Section | Description |
|-|-|
| [Introduction](#Introduction) | Introduction to CINO |
| [Download](#Download) | Download links and how-to-use |
| [Quick Load](#Quick-Load) | Learn how to quickly load our models through [🤗Transformers](https://github.com/huggingface/transformers) |
| [Dataset for Chinese Minority Languages](#Dataset-for-Chinese-Minority-Languages) | Introduce Wiki-Chinese-Minority (WCM) and other datasets |
| [Results](#Results) | Results on several datasets |


## Introduction
Multilingual Pre-trained Language Model, such as mBERT and XLM-R, adopts masked language model (MLM) and other self-supervised approaches to support multilingual and cross-lingual abilities in NLP systems, using training corpus in various languages.

However, due to the scarcity of corpus in Chinese minority languages and neglection of relevant research, current multilingual PLMs are not capable of dealing with these languages.

We made the following contributions.

- We propose **CINO** (**Ch**inese m**INO**rity PLM), which is built on [XLM-R](https://github.com/facebookresearch/XLM). We further pre-train XLM-R with corpus in Chinese minority languages. 

- To evaluate CINO as well as other multilingual PLMs, we also propose a new classification dataset called **Wiki-Chinese-Minority（WCM）**, which is built on Wikipedia.

- The experimental results on WCM, Tibetan News Classification Corpus (TNCC),  and KLUE-TC (YNAT) show that CINO achieves state-of-the-art performances.

CINO supports the following languages:

- Chinese，中文（zh）
- Tibetan，藏语（bo）
- Mongolian (Uighur form)，蒙语（mn）
- Uyghur，维吾尔语（ug）
- Kazakh (Arabic form)，哈萨克语（kk）
- Korean，朝鲜语（ko）
- Zhuang，壮语
- Cantonese，粤语（yue）

<p align="center">
    <br>
    <img src="./pics/chinese_minority_model.png" width="1000"/>
    <br>
</p>


## Download

### Direct Download

We provide CINO-base and CINO-large of PyTorch version (**preferred version: v2**). We will release more models in the future.

* **`CINO-large-v2`**：24-layer, 1024-hidden, 16-heads, vocabulary size 136K, 442M parameters
* **`CINO-base-v2`** 12-layer, 768-hidden, 12-heads, vocabulary size 136K, 190M parameters
* **`CINO-large`**：24-layer, 1024-hidden, 16-heads, vocabulary size 275K, 585M parameters

Notice:

* v1 model（CINO-large）supports all the languages in XLM-R and the minority languages.
* v2 models (CINO-base-v2 and CINO-large-v2) have pruned vocabularies and only support Chinese and the minority languages.


| Model | Size | Google Drive | Baidu Disk |
| :------- | :---------: |  :---------: |  :---------: |
| **CINO-large-v2** | **1.6GB** | **[PyTorch](https://drive.google.com/file/d/13ehEZcAJmW_19IOiKOJX49tfmWclqn6_/view?usp=sharing)** | **[PyTorch（pw: 3fjt）](https://pan.baidu.com/s/19wks3DpI2gXxAD8twN12Jg?pwd=3fjt)** |
| **CINO-base-v2** | **705MB** | **[PyTorch](https://drive.google.com/file/d/1DxTPFG3DYIF_TyoFQw1-lJ_xuQipSnLd/view?usp=sharing)** | **[PyTorch（pw: qnvc）](https://pan.baidu.com/s/11qOk7YaGRsJJl3QviNR0IA?pwd=qnvc)** |
| **CINO-large** | **2.2GB** | **[PyTorch](https://drive.google.com/file/d/1-79q1xLXG2QQ4cdoemiRQVlWrNNRcZl2/view?usp=sharing)** | **[PyTorch (pw: wpyh)](https://pan.baidu.com/s/1xOsUbwwY1K6rMysEvGXSLg?pwd=wpyh)** |

### Download from 🤗transformers

You can also download our models from 🤗transformers Model Hub, including PyTorch and Tensorflow2 models.
| Model | Size | transformers model hub URL |
| :------- | :---------: |  :---------: |
| **CINO-large-v2** | **1.6GB** | https://huggingface.co/hfl/cino-large-v2 |
| **CINO-base-v2** | **705MB** | https://huggingface.co/hfl/cino-base-v2 |
| **CINO-large** | **2.2GB** | https://huggingface.co/hfl/cino-large |

How-to: click the model link that you wish to download (e.g., https://huggingface.co/hfl/cino-large) → Select "Files and versions" tab  → Download!

### How-To-Use

There are three files in PyTorch model:

```
pytorch_model.bin        # Model Weight
config.json              # Model Config
sentencepiece.bpe.model  # Vocabulary
```
CINO uses exactly the same neural architecture with XLM-R, which can be direclty loaded using `XLMRobertaModel` class in [Transformers](https://huggingface.co/transformers/).

```
from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained("PATH_TO_MODEL_DIR")
model = XLMRobertaModel.from_pretrained("PATH_TO_MODEL_DIR")
```

## Quick Load
With [🤗Transformers](https://github.com/huggingface/transformers), the models above could be easily accessed and loaded through the following codes.

```
from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained("MODEL_NAME")
model = XLMRobertaModel.from_pretrained("MODEL_NAME")
```

The actual model and its `MODEL_NAME` are listed below.

| Actual Model | MODEL_NAME |
| - | - |
| CINO-large-v2 | hfl/cino-large-v2 |
| CINO-base-v2 | hfl/cino-base-v2 |
| CINO-large | hfl/cino-large |

## Dataset for Chinese Minority Languages

### Wiki-Chinese-Minority（WCM）
We built a new classification dataset **Wiki-Chinese-Minority (WCM)**. The dataset covers Mongolian, Tibetan, Uyghur, Cantonese, Korean, Kazakh, and Chinese, including ten categories of art, geography, history, nature, natural science, people, technology, education, economy, and health.

We use [weighted-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for evaluation. 

| Name | Google Drive | Baidu Disk |
| :------- |  :---------: |  :---------: |
| **Wiki-Chinese-Minority-v2（WCM-v2）** | [Google Drive](https://drive.google.com/file/d/1wlCs_6YAfygPPVwXPuiUZ_uqs1BgFODX/view?usp=sharing) |[Baidu Disk (pw: sbse)](https://pan.baidu.com/s/1X7Q-K9rfDyEdDp81G2lxOw?pwd=sbse) |
| **Wiki-Chinese-Minority（WCM）** | [Google Drive](https://drive.google.com/file/d/1VuP_inhluxq7d71xjHSYRRncIwWgjy_L/view?usp=sharing) | [Baidu Disk (pw: y9sw)](https://pan.baidu.com/s/1fZ8ZgPezFxNFdUYchpxfgQ?pwd=y9sw) |

WCM-v2 has a more balanced data distribution across categories and languages.

Dataset Statistics of WCM-v2:

| Category | mn | bo | ug | yue | ko | Kk | zh-Train | zh-Dev | zh-Test |
| :------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| Art |  135  | 141 |3|387|806|   348   |2657|331|335|
| Geography | 76 | 339 |256|1550|1197|572|12854|1589|1644|
| History | 66 | 111 |0|499|776|491|1771|227|248|
| Nature | 7    | 0 | 7 | 606 | 442 | 361 |  1105 | 134 |  110 |
| Natural Science | 779    | 133 | 20  | 336 | 532 | 880 |    2314    |    317    | 287   |
| People | 1402 | 111 | 0 | 1230 | 684  | 169 | 7706 | 953 |  924 |
| Technology | 191  | 163 | 8 | 329 | 808 | 515 |   1184 | 134 |  152 |
| Education | 6    | 1 | 0 | 289  | 439 | 1392 | 936   | 130 | 118   |
| Economy | 205 | 0  | 0 | 445 | 575 | 637 | 922   | 113 | 109   |
| Health | 106  | 111  | 6 | 272  | 299  | 893 | 551   | 67 | 73    |
| **Total** | 2973 | 1110 | 300 | 5943 | 6558 | 6258 | 32000 | 3995 | 4000 |

Note:

* The dataset includes two folders: `zh` and `minority`
* zh: train/dev/test in Chinese
* minority: test set for all languages

**The dataset is still in its alpha stage, with possible modifications in the future.**

## Results

We evaluate on YNAT, TNCC, and Wiki-Chinese-Minority. For each dataset, we use the same hyper-params for all models.

### Korean Text Classification (YNAT)
* **KLUE-TC (a.k.a. YNAT)** is released by KLUE.
* Dataset Source: [KLUE benchmark](https://klue-benchmark.com)
* Dataset Details: [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680.pdf)

| #Train | #Dev  | #Test | #Classes | Metric   |
| :------: |:------: | :------: | :------: | :------: |
| 45,678 | 9,107 | 9,107 | 7        | macro-F1 |

Hyper-params: Initial LR1e-5, batch size 16.

Results:

| Model | Dev |
| :------- | :-----: |
| XLM-R-large<sup>[1]</sup> | 87.3 |
| XLM-R-large<sup>[2]</sup> | 86.3 |
| **CINO-base-v2** | 85.5 |
| **CINO-large-v2** | 87.2 |
| **CINO-large** | **87.4** |

 > [1] The results in the original paper.  
 > [2] Reproduced result using the same initial LR with CINO-large.


### Tibetan News Classification Corpus（TNCC）
* **Tibetan News Classification Corpus (TNCC)** is released by Fudan University.
* Dataset Source: [Tibetan-Classification](https://github.com/FudanNLP/Tibetan-Classification)
* Details of dataset: [End-to-End Neural Text Classification for Tibetan](http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-104.pdf)

| #Train<sup>[1]</sup> | #Dev | #Test | #Classes | Metric   |
| :----: | :----: | :----: | :----: | :----: |
|  7,363  | 920 | 920  | 12        | macro-F1 |

Hyper-params:  initial LR 5e-6, batch size 16

Results:

| Model | Dev | Test |
| :------- | :---------: | :---------: |
| TextCNN | 65.1 | 63.4 |
| XLM-R-large | 14.3 | 13.3 |
| **CINO-base-v2** | 71.0 | 69.3 |
| **CINO-large-v2** | **72.9** | **71.0** |
| **CINO-large** | 71.3 | 68.6 |


> Note: there is no official train/dev/test split in this dataset. We split the dataset with the ratio of 8:1:1.

### Wiki-Chinese-Minority

We use Chinese training set to train our model and test on other languages (zero-shot). We use weighted-F1 for evaluation.

Hyper-params: initial LR 7e-6, batch size 32.

Results on WCM-v2:

| Model | MN | BO | UG | YUE | KO | KK | ZH | Average |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | ----------- |
| XLM-R-base |  41.2  | 25.7 |   84.5   | 66.1 |  43.1  |   23.0   | 88.3 | 53.1      |
| XLM-R-large |  53.8  | 24.5 |   89.4   | 67.3 |  45.4  |   30.0   | 88.3 | 57.0     |
| CINO-base-v2   |  62.1  | 52.7 |   87.8   | 68.1 |  45.6  |   38.3   | 89.0 | 63.4     |
| CINO-large-v2 | 73.1 | 58.9 |   90.1   |66.9|45.1|   42.0   |88.9|**66.4**|

## Demo Code

See `examples`. It currently includes

* [examples/WCM](examples/WCM/README_EN.md)：Fine-tuning and zero-shot evaluation on WCM
* [examples/TNCC](examples/TNCC/README_EN.md)：Fine-tuning on TNCC
* [examples/YNAT](examples/YNAT/README_EN.md)：Fine-tuning on YNAT


## Technical Details

TBA

## Follow Us
Follow our official WeChat account to keep updated with our latest technologies!

![qrcode.jpg](pics/qrcode.jpg)

## Issues
Please submit an issue.
