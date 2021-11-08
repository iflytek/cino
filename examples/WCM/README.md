## 使用方法

[**中文说明**](README.md) | [**English**](README_EN.md)

用户可以基于已发布的上述预训练模型进行下游任务精调及zero-shot测试。

### 测试环境
- numpy : 1.21.2
- python : 3.7.10
- pytorch : 1.7.1
- scikit-learn : 0.24.2
- transformers : 3.1.0

### Finetune

本例中，我们使用 `CINO-large` 模型在维基分类任务中文数据上进行精调，相关步骤如下。假设，  
- `project-dir`：工作根目录，可按实际情况设置。
- `data-dir`：数据目录，本例为 `${project-dir}/data/`。
- `model_pretrain_dir`：预训练模型目录，本例为 `${project-dir}/model/`。
- `model_save_dir`：精调最优模型参数存储目录，本例为 `${project-dir}/saved_models/`。
- `best_model_save_name`：精调最优模型参数文件名，本例为 `best_cino.pth`。

#### 第一步：模型准备
在[模型下载](https://github.com/ymcui/Chinese-Minority-PLM#模型下载)章节中，下载`CINO-large`模型，并解压至`${project-dir}/model/`。
该目录下应包含`pytorch_model.bin`，`sentencepiece.bpe.model`，`config.json`，共计3个文件。

#### 第二步：数据准备
参照[少数民族语言分类数据集](https://github.com/ymcui/Chinese-Minority-PLM#%E5%B0%91%E6%95%B0%E6%B0%91%E6%97%8F%E8%AF%AD%E8%A8%80%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86)章节中的说明，下载Wiki-Chinese-Minority（WCM）数据集中中文数据到`${data-dir}`，并保持原文件名。即`train.txt`、`dev.txt`和`test.txt`。

#### 第三步：运行训练命令
```shell
python run_finetune.py --params cino-params.json
```
`params`是一个JSON词典，在本例中的`cino-params.json`包含了精调相关参数，例如：
```json
{
    "learning_rate":5e-6,
    "epoch":5,
    "gradient_acc":4,
    "batch_size":16,
    "max_len":512,
    "weight_decay":1e-4,
    "warmup_rate":0.1,
    "data_dir":"data/",
    "model_pretrain_dir":"model/", 
    "model_save_dir":"saved_models/",
    "best_model_save_name":"best_cino.pth",
    "class_names":["艺术", "地理", "历史", "自然", "自然科学", "人物", "技术", "教育", "经济", "健康"] 
}
```

运行完毕后，精调过程的日志信息和模型测试结果可在`${project-dir}/log/cino-ft.log`中查看。



### Zero-Shot

本例中，我们使用 `CINO-large` 模型在维基分类任务少数民族语言数据上进行zero-shot测试，相关步骤如下。假设，  
- `project-dir`：工作根目录，可按实际情况设置。
- `data-dir`：数据目录，本例为 `${project-dir}/data/`。
- `model_pretrain_dir`：预训练模型目录，本例为 `${project-dir}/model/`。
- `model_finetune_params`：用于zero-shot的模型参数路径，本例为 `${project-dir}/model/best_cino.pth`。

#### 第一步：模型准备
在[模型下载](https://github.com/ymcui/Chinese-Minority-PLM#模型下载)章节中，下载`CINO-large`模型，并解压至`${project-dir}/model/`。
该目录下应包含`pytorch_model.bin`，`sentencepiece.bpe.model`，`config.json`，共计3个文件。

将finetune阶段保存的最优模型参数文件放于`${project-dir}/model/`。

#### 第二步：数据准备
参照[少数民族语言分类数据集](https://github.com/ymcui/Chinese-Minority-PLM#%E5%B0%91%E6%95%B0%E6%B0%91%E6%97%8F%E8%AF%AD%E8%A8%80%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86)章节中的说明，下载Wiki-Chinese-Minority（WCM）数据集中少数民族语言数据到`${data-dir}`，并保持原文件名。即`bo.txt`、`kk.txt`、`ko.txt`、`mn.txt`、`ug.txt`和`yue.txt`。

#### 第三步：运行训练命令
```shell
python run_zeroshot.py --params cino-params.json
```
`params`是一个JSON词典，在本例中的`cino-params.json`包含了zero-shot相关参数，例如：
```json
{
    "batch_size":16,
    "max_len":512,
    "model_pretrain_dir":"model/",
    "model_finetune_params":"model/best_cino.pth",
    "data_dir":"data/",
    "class_names":["艺术", "地理", "历史", "自然", "自然科学", "人物", "技术", "教育", "经济", "健康"] 
}
```

运行完毕后，zero-shot测试结果可在`${project-dir}/log/cino-zs.log`中查看。

