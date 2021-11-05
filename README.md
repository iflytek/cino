# 少数民族语言预训练模型（Chinese-Minority-PLM）
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


其他相关资源：
- 中文MacBERT预训练模型：https://github.com/ymcui/MacBERT
- 中文ELECTRA预训练模型：https://github.com/ymcui/Chinese-ELECTRA
- 中文BERT-wwm预训练模型：https://github.com/ymcui/Chinese-BERT-wwm
- 中文XLNet预训练模型：https://github.com/ymcui/Chinese-XLNet
- 知识蒸馏工具TextBrewer：https://github.com/airaria/TextBrewer

查看更多哈工大讯飞联合实验室（HFL）发布的资源：https://github.com/ymcui/HFL-Anthology

## 新闻

2021/10/25 **CINO-large模型、少数民族语言分类任务数据集Wiki-Chinese-Minority（WCM）数据集已开放下载使用。**



## 内容导引

| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍少数民族语言预训练模型与相关数据集 |
| [模型下载](#模型下载) | 模型下载地址与使用说明 |
| [快速加载](#快速加载) | 介绍了如何使用[🤗Transformers](https://github.com/huggingface/transformers)快速加载模型 |
| [少数民族语言分类数据集](#少数民族语言分类数据集) | 介绍少数民族语言分类数据集 |
| [实验结果](#实验结果) | 列举了模型在NLU任务上的效果 |
| [技术细节](#模型细节与技术报告) | 描述模型的技术细节 |


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

目前提供PyTorch版本的CINO-large模型的下载，后续将陆续更新其他规模与版本的模型。

* **`CINO-large`**：24-layer, 1024-hidden, 16-heads, 585M parameters  

| 模型简称 | 模型文件大小 | Google下载 |  讯飞云下载 |
| :------- | :---------: |  :---------: |  ----------- |
| **CINO-large** | **2.2GB** | **[PyTorch模型](https://drive.google.com/file/d/1-79q1xLXG2QQ4cdoemiRQVlWrNNRcZl2/view?usp=sharing)** |  **[PyTorch模型（密码buhD）](http://pan.iflytek.com:80/link/5D942296A74C97F9FD68E509D1C934EC)** |

### 通过🤗transformers下载

通过🤗transformers模型库可以下载TensorFlow和PyTorch版本模型。
| 模型简称 | 模型文件大小 | transformers模型库地址 |
| :------- | :---------: |  :---------: |
| **CINO-large** | **2.2GB** | https://huggingface.co/hfl/cino-large |

下载方法：点击任意需要下载的模型 → 选择"Files and versions"选项卡 → 下载对应的模型文件。

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
| CINO-large | hfl/cino-large |

## 少数民族语言分类数据集

### Wiki-Chinese-Minority（WCM）
我们基于少数民族语言维基百科语料及其分类体系标签，构建了分类任务数据集 **Wiki-Chinese-Minority（WCM）**。该数据集覆盖了蒙古语、藏语、维吾尔语、粤语、朝鲜语、哈萨克语，中文，包括艺术、地理、历史、自然、自然科学、人物、技术、教育、经济和健康十个类别。

各个语言上取[weighted-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)为评测指标。计算所有语言的weighted-F1平均作为总体评价指标。

| 数据集名称 | Google下载 |  讯飞云下载 |
| :------- |  :---------: |  ----------- |
| **Wiki-Chinese-Minority（WCM）** | [Google Drive](https://drive.google.com/file/d/1VuP_inhluxq7d71xjHSYRRncIwWgjy_L/view?usp=sharing) |   **[（密码UW4s）](http://pan.iflytek.com:80/link/EE3D3364E2E66489395130CDF7930818)** |

数据集分布：

| 类别 | 蒙古语 | 藏语 | 维吾尔语 | 粤语 | 朝鲜语 | 哈萨克语 |  中文-Train | 中文-Dev | 中文-Test |
| :------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| 艺术 |  437  | 129 |3|3877|10962|   802   |2657|335|331|
| 地理 | 145 | 0 |256|30488|17121|347|12854|1644|1589|
| 历史 | 470 | 125 |0|6434|10491|588|1771|248|227|
| 自然 | 90	| 0	| 7	| 8880	| 5250 |	5050 |	1105 |	110 |	134 |
| 自然科学 | 5366	| 129 |	20	| 3365	| 6654	| 4183 |	2314	| 287	| 317 |
| 人物 | 428 |	0 |	0 |	23163 |	9057	| 870 |	7706 |	924 |	953 |
| 技术 | 112	| 1	| 8	| 3293 |	10997 |	635 |	1184 |	152 |	134 |
| 教育 | 212	| 0	| 0	| 2892	| 5193	| 13973	| 936	| 118	| 130 |
| 经济 | 0 | 0	| 0	| 5192	| 7343	| 2712	| 922	| 109	| 113 |
| 健康 | 0	| 110	| 6	| 2721	| 2994	| 2176	| 551	| 73	| 67 |
| **总计** | 7260 | 494 | 300 | 90305 | 86062 | 31336 | 32000 | 4000 | 3995 |

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
| **CINO-large** | **71.3** | **68.6** |

> 注：原论文中未提供train/dev/test的划分方式。因此，我们重新对数据集按8:1:1做了划分。

### Wiki-Chinese-Minority

在中文训练集上训练，在其他语言上做zero-shot测试。各语言的评测指标为weighted-F1。

实验参数：学习率为7e-6，batch_size为32。

实验结果：

| 模型 | 蒙古语 | 藏语 | 维吾尔语 | 粤语 | 朝鲜语 | 哈萨克语 |  中文 | Average |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | ----------- |
| XLM-R-large |  **33.2**  | 22.9 |77.4|71.4|44.2|   11.6   |88.4|49.9|
| **CINO-large** | 20.0 | **31.5** |**88.8**|**72.3**|**46.2**|**26.1**|**89.6**|**53.5**|

## 示例代码

请参考[examples](https://github.com/GeekDream-x/Chinese-Minority-PLM/tree/main/examples)。


## 模型细节与技术报告

将在近期公布，敬请期待。

## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

![qrcode.jpg](pics/qrcode.jpg)


## 问题反馈
如有问题，请在GitHub Issue中提交。

