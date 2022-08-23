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

在自然语言处理领域中，预训练语言模型（Pre-trained Language Model, PLM）已成为重要的基础技术，在多语言的研究中，预训练模型的使用也愈加普遍。为了促进中国少数民族语言信息处理的研究与发展，**哈工大讯飞联合实验室（HFL）**发布少数民族语言预训练模型**CINO** (**Ch**inese m**INO**rity PLM)。  

----

[中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) |  [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

查看更多哈工大讯飞联合实验室发布的资源：https://github.com/ymcui/HFL-Anthology

## 新闻
**2022/8/23 CINO被国际重要会议[COLING 2022](http://coling2022.org)录用为长文。camera-ready结束后，我们将更新论文最终版并发布相应资源。**

2022/3/30 我们开源了一种新预训练模型PERT。查看：https://github.com/ymcui/PERT**

2022/02/21 更新CINO-small模型，6层transformer结构，参数量148M。

2022/01/25 更新CINO-v2模型与WCM-v2数据集，少数民族语言分类任务效果提升。

2021/12/17 哈工大讯飞联合实验室全新推出[模型裁剪工具包TextPruner](https://github.com/airaria/TextPruner)，欢迎试用。

2021/10/25 CINO-large模型、少数民族语言分类任务数据集Wiki-Chinese-Minority（WCM）数据集已开放下载使用。


## 内容导引

| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍少数民族语言预训练模型与相关数据集 |
| [模型下载](#模型下载) | 模型下载地址与使用说明 |
| [快速加载](#快速加载) | 介绍了如何使用[🤗Transformers](https://github.com/huggingface/transformers)快速加载模型 |
| [少数民族语言分类数据集](#少数民族语言分类数据集) | 介绍少数民族语言分类数据集 |
| [实验结果](#实验结果) | 列举了模型在NLU任务上的效果 |
| [引用](#引用) | 技术报告与引用 |


## 简介
多语言预训练模型（Multilingual Pre-trained Language Model），如mBERT、XLM-R等，通过在预训练阶段增加语言数量、采用MLM自监督训练等方式，使预训练模型具备了多语言（multilingual）和跨语言（cross-lingual）理解的能力。然而，由于国内少数民族语言语料的稀缺以及国际上研究的忽视，现有的多语言模型无法很好地处理国内少数民族语言文字。

本项工作的主要贡献：

- **CINO** (**Ch**inese m**INO**rity PLM) 基于多语言预训练模型[XLM-R](https://github.com/facebookresearch/XLM)，在多种国内少数民族语言语料上进行了二次预训练。该模型提供了藏语、蒙语（回鹘体）、维吾尔语、哈萨克语（阿拉伯体）、朝鲜语、壮语、粤语等少数民族语言与方言的理解能力。

- 为了便于评价包括CINO在内的各个多语言预训练模型性能，我们构建了基于维基百科的少数民族语言分类任务数据集**Wiki-Chinese-Minority（WCM）**。具体见[少数民族语言分类数据集](#少数民族语言分类数据集)。

- 通过实验证明，CINO在Wiki-Chinese-Minority（WCM）以及其他少数民族语言数据集：藏语新闻分类 Tibetan News Classification Corpus (TNCC) 、朝鲜语新闻分类 KLUE-TC (YNAT) 上获得了最好的效果。相关结果详见[实验结果](#实验结果)。

该模型涵盖：

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


## 模型下载

### 直接下载


目前提供PyTorch版本的CINO-small、CINO-base和CINO-large模型的下载（**推荐使用v2版本**），后续将陆续更新其他规模与版本的模型。

* **`CINO-large-v2`**：24-layer, 1024-hidden, 16-heads, vocabulary size 136K, 442M parameters
* **`CINO-base-v2`** 12-layer, 768-hidden, 12-heads, vocabulary size 136K, 190M parameters
* **`CINO-small-v2`** 6-layer, 768-hidden, 12-heads, vocabulary size 136K, 148M parameters
* **`CINO-large`**：24-layer, 1024-hidden, 16-heads, vocabulary size 275K, 585M parameters

注意：
- v1模型（CINO-large）支持XLM-R中的所有语言再加上少数民族语言；
- v2模型（CINO-large-v2，CINO-base-v2和CINO-small-v2）的词表针对预训练数据做了裁剪，仅支持中文与少数民族语言。


| 模型简称 | 模型文件大小 | Google下载 | 百度网盘下载 |
| :------- | :---------: |  :---------: |  :---------: |
| **CINO-large-v2** | **1.6GB** | **[PyTorch模型](https://drive.google.com/file/d/13ehEZcAJmW_19IOiKOJX49tfmWclqn6_/view?usp=sharing)** | **[PyTorch模型（密码3fjt）](https://pan.baidu.com/s/19wks3DpI2gXxAD8twN12Jg?pwd=3fjt)** |
| **CINO-base-v2** | **705MB** | **[PyTorch模型](https://drive.google.com/file/d/1DxTPFG3DYIF_TyoFQw1-lJ_xuQipSnLd/view?usp=sharing)** | **[PyTorch模型（密码qnvc）](https://pan.baidu.com/s/11qOk7YaGRsJJl3QviNR0IA?pwd=qnvc)** |
| **CINO-small-v2** | **564MB** | **[PyTorch模型](https://drive.google.com/file/d/1cEVoehOEdNJdHfJ-lUXGvVVQ8vdxYs__/view?usp=sharing)** | **[PyTorch模型（密码9mc8）](https://pan.baidu.com/s/1tC_doYl6pxvJpfyIDVTCQg?pwd=9mc8)** |
| **CINO-large** | **2.2GB** | **[PyTorch模型](https://drive.google.com/file/d/1-79q1xLXG2QQ4cdoemiRQVlWrNNRcZl2/view?usp=sharing)** | **[PyTorch模型（密码wpyh）](https://pan.baidu.com/s/1xOsUbwwY1K6rMysEvGXSLg?pwd=wpyh)** |

### 通过🤗transformers下载

通过🤗transformers模型库可以下载TensorFlow (v2)和PyTorch版本模型。

下载方法：点击任意需要下载的模型 → 选择"Files and versions"选项卡 → 下载对应的模型文件。

| 模型简称 | 模型文件大小 | transformers模型库地址 |
| :------- | :---------: |  :---------: |
| **CINO-large-v2** | **1.6GB** | https://huggingface.co/hfl/cino-large-v2 |
| **CINO-base-v2** | **705MB** | https://huggingface.co/hfl/cino-base-v2 |
| **CINO-small-v2** | **564MB** | https://huggingface.co/hfl/cino-small-v2 |
| **CINO-large** | **2.2GB** | https://huggingface.co/hfl/cino-large |

### 模型使用

PyTorch版本包含3个文件：
```
pytorch_model.bin        # 模型权重
config.json              # 模型参数
sentencepiece.bpe.model  # 词表
```
CINO的结构与XLM-R相同，可直接使用[Transformers](https://huggingface.co/transformers/)中的`XLMRobertaModel`模型进行加载：
```
from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained("PATH_TO_MODEL_DIR")
model = XLMRobertaModel.from_pretrained("PATH_TO_MODEL_DIR")
```

## 快速加载
依托于[🤗Transformers](https://github.com/huggingface/transformers)，可轻松调用以上CINO模型。
```
from transformers import XLMRobertaTokenizer, XLMRobertaModel
tokenizer = XLMRobertaTokenizer.from_pretrained("MODEL_NAME")
model = XLMRobertaModel.from_pretrained("MODEL_NAME")
```

其中`MODEL_NAME`对应列表如下：

| 模型名 | MODEL_NAME |
| - | - |
| CINO-large-v2 | hfl/cino-large-v2 |
| CINO-base-v2 | hfl/cino-base-v2 |
| CINO-small-v2 | hfl/cino-small-v2 |
| CINO-large | hfl/cino-large |

## 少数民族语言分类数据集

### Wiki-Chinese-Minority（WCM）
我们基于少数民族语言维基百科语料及其分类体系标签，构建了分类任务数据集 **Wiki-Chinese-Minority（WCM）**。该数据集覆盖了蒙古语、藏语、维吾尔语、粤语、朝鲜语、哈萨克语，中文，包括艺术、地理、历史、自然、自然科学、人物、技术、教育、经济和健康十个类别。

各个语言上取[weighted-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)为评测指标。计算所有语言的weighted-F1平均作为总体评价指标。

| 数据集名称 | Google下载 | 百度网盘下载 |
| :------- |  :---------: |  :---------: |
| **Wiki-Chinese-Minority-v2（WCM-v2）** | [Google Drive](https://drive.google.com/file/d/1wlCs_6YAfygPPVwXPuiUZ_uqs1BgFODX/view?usp=sharing) |[数据集（密码sbse）](https://pan.baidu.com/s/1X7Q-K9rfDyEdDp81G2lxOw?pwd=sbse) |
| **Wiki-Chinese-Minority（WCM）** | [Google Drive](https://drive.google.com/file/d/1VuP_inhluxq7d71xjHSYRRncIwWgjy_L/view?usp=sharing) | [数据集（密码y9sw）](https://pan.baidu.com/s/1fZ8ZgPezFxNFdUYchpxfgQ?pwd=y9sw) |

**WCM-v2版本调整了各类别与语言的样本数量，分布相对更均衡**

WCM-v2版本数据分布：

| 类别 | 蒙古语 | 藏语 | 维吾尔语 | 粤语 | 朝鲜语 | 哈萨克语 |  中文-Train | 中文-Dev | 中文-Test |
| :------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| 艺术 |  135  | 141 |3|387|806|   348   |2657|331|335|
| 地理 | 76 | 339 |256|1550|1197|572|12854|1589|1644|
| 历史 | 66 | 111 |0|499|776|491|1771|227|248|
| 自然 | 7    | 0 | 7 | 606 | 442 | 361 |  1105 | 134 |  110 |
| 自然科学 | 779    | 133 | 20  | 336 | 532 | 880 |    2314    |    317    | 287   |
| 人物 | 1402 | 111 | 0 | 1230 | 684  | 169 | 7706 | 953 |  924 |
| 技术 | 191  | 163 | 8 | 329 | 808 | 515 |   1184 | 134 |  152 |
| 教育 | 6    | 1 | 0 | 289  | 439 | 1392 | 936   | 130 | 118   |
| 经济 | 205 | 0  | 0 | 445 | 575 | 637 | 922   | 113 | 109   |
| 健康 | 106  | 111  | 6 | 272  | 299  | 893 | 551   | 67 | 73    |
| **总计** | 2973 | 1110 | 300 | 5943 | 6558 | 6258 | 32000 | 3995 | 4000 |

数据说明：

* 包含两个文件夹：zh和minority
* zh：中文的训练集、开发集和测试集
* minority：所有语言（各少数民族语言与方言）的测试集

**该数据集尚处于alpha阶段，之后的版本可能会有一定改动。**  
**后续还将有其他数据集发布，敬请期待。**

## 实验结果

我们在YNAT、TNCC和Wiki-Chinese-Minority三个数据集上比较了不同模型的效果。

对于同一任务上的各个预训练模型，使用统一的训练轮数、学习率等参数。

### 朝鲜语文本分类（YNAT）
* 该任务选用由KLUE团队发布的朝鲜语新闻数据集**KLUE-TC (a.k.a. YNAT)**
* 数据集来源：[KLUE benchmark](https://klue-benchmark.com)
* 详细信息参阅论文：[KLUE: Korean Language Understanding Evaluation](https://arxiv.org/pdf/2105.09680.pdf)

| #Train | #Dev  | #Test | #Classes | Metric   |
| :------: |:------: | :------: | :------: | :------: |
| 45,678 | 9,107 | 9,107 | 7        | macro-F1 |

实验参数：学习率为1e-5，batch_size为16。

实验结果：

| 模型 | 开发集 |
| :------- | :-----: |
| XLM-R-large<sup>[1]</sup> | 87.3 |
| XLM-R-large<sup>[2]</sup> | 86.3 |
| **CINO-small-v2** |84.1 |
| **CINO-base-v2** | 85.5 |
| **CINO-large-v2** | 87.2 |
| **CINO-large** | **87.4** |

 > [1] 论文中的结果。  
 > [2] 复现结果，与CINO-large使用相同的学习率。


### 藏语文本分类（TNCC）
* 该任务选用由复旦大学自然语言处理实验室发布的藏语新闻数据集 **Tibetan News Classification Corpus (TNCC)**
* 数据集来源：[Tibetan-Classification](https://github.com/FudanNLP/Tibetan-Classification)
* 详细信息参阅论文：[End-to-End Neural Text Classification for Tibetan](http://www.cips-cl.org/static/anthology/CCL-2017/CCL-17-104.pdf)

| #Train<sup>[1]</sup> | #Dev | #Test | #Classes | Metric   |
| :----: | :----: | :----: | :----: | :----: |
|  7,363  | 920 | 920  | 12        | macro-F1 |

实验参数：学习率为5e-6，batch_size为16。

实验结果：

| 模型 | 开发集 | 测试集 |
| :------- | :---------: | :---------: |
| TextCNN | 65.1 | 63.4 |
| XLM-R-large | 14.3 | 13.3 |
| **CINO-small-v2** | 72.1 | 66.7 |
| **CINO-base-v2** | 70.3 | 68.4 |
| **CINO-large-v2** | **72.9** | **71.0** |
| **CINO-large** | 71.3 | 68.6 |

> 注：原论文中未提供train/dev/test的划分方式。因此，我们重新对数据集按8:1:1做了划分。

### Wiki-Chinese-Minority

在中文训练集上训练，在其他语言上做zero-shot测试。各语言的评测指标为weighted-F1。

实验参数：学习率为7e-6，batch_size为32。

WCM-v2实验结果：

| 模型 | 蒙古语 | 藏语 | 维吾尔语 | 粤语 | 朝鲜语 | 哈萨克语 |  中文 | Average |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | ----------- |
| XLM-R-base |  41.2  | 25.7 |   84.5   | 66.1 |  43.1  |   23.0   | 88.3 | 53.1      |
| XLM-R-large |  53.8  | 24.5 |   89.4   | 67.3 |  45.4  |   30.0   | 88.3 | 57.0     |
| CINO-small-v2   | 60.3  | 47.9 |   86.5   | 64.6 |  43.2  |   33.2   | 87.9 | 60.5         |
| CINO-base-v2   |  62.1  | 52.7 |   87.8   | 68.1 |  45.6  |   38.3   | 89.0 | 63.4     |
| CINO-large-v2 | 73.1 | 58.9 |   90.1   |66.9|45.1|   42.0   |88.9|**66.4**|

## 示例代码

参见`examples`目录，目前包括

* [examples/WCM](examples/WCM/README.md)：WCM上的精调与zero-shot测试
* [examples/TNCC](examples/TNCC/README.md)：TNCC上的精调
* [examples/YNAT](examples/YNAT/README.md)：YNAT上的精调

## 引用

如果本目录中的内容对你的研究工作有所帮助，欢迎引用下述论文。

- [CINO: A Chinese Minority Pre-trained Language Model](https://arxiv.org/abs/2202.13558)
```
@misc{yang2022cino,
      title={CINO: A Chinese Minority Pre-trained Language Model}, 
      author={Ziqing Yang and Zihang Xu and Yiming Cui and Baoxin Wang and Min Lin and Dayong Wu and Zhigang Chen},
      year={2022},
      eprint={2202.13558},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

![qrcode.jpg](pics/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

