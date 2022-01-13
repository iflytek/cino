## 使用方法

[**中文说明**](README.md) | [**English**](README_EN.md)

用户可以基于已发布的预训练模型CINO在YNAT朝鲜语数据集上进行精调实验。

### 测试环境
- numpy : 1.21.2
- python : 3.7.10
- pytorch : 1.7.1
- scikit-learn : 0.24.2
- transformers : 3.1.0

### 示例步骤

本例中，我们使用 `CINO-large` 模型在YNAT朝鲜语数据集上进行精调，相关步骤如下。假设，  
- `project-dir`：工作根目录，可按实际情况设置。
- `data-dir`：数据目录，本例为 `${project-dir}/data/`。
- `model_pretrain_dir`：预训练模型目录，本例为 `${project-dir}/model/`。
- `model_save_dir`：精调最优模型参数存储目录，本例为 `${project-dir}/saved_models/`。
- `best_model_save_name`：精调最优模型参数文件名，本例为 `best_cino.pth`。

#### 第一步：模型准备
在[模型下载](https://github.com/ymcui/Chinese-Minority-PLM#模型下载)章节中，下载`CINO-large`模型，并解压至`${project-dir}/model/`。
该目录下应包含`pytorch_model.bin`，`sentencepiece.bpe.model`，`config.json`，共计3个文件。

#### 第二步：数据准备
参照[朝鲜语文本分类（YNAT）](https://github.com/ymcui/Chinese-Minority-PLM/#%E6%9C%9D%E9%B2%9C%E8%AF%AD%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BBynat)章节中的说明，下载数据集到`${data-dir}`，并将训练集和开发集分别重命名为`train.txt`和`dev.txt`。

#### 第三步：运行训练命令
```shell
python ynat_finetune.py --params cino-params.json
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
    "class_names":["政治", "经济", "社会", "文化", "世界", "IT/科学", "运动"]
}
```

运行完毕后，精调过程的日志信息和模型在开发集的测试结果可在`${project-dir}/log/cino_ynat.log`中查看。

