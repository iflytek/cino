## Usage

[**中文说明**](README.md) | [**English**](README_EN.md)

This folder contains examples for finetuning our pretrained model CINO with Korean dataset - YNAT.

### Requirements
- numpy : 1.21.2
- python : 3.7.10
- pytorch : 1.7.1
- scikit-learn : 0.24.2
- transformers : 3.1.0

### Steps

In this tutorial, we will finetune `CINO-large` with Korean dataset - YNAT.
- `project-dir`：working directory
- `data-dir`：data directory, here we set as `${project-dir}/data/`
- `model_pretrain_dir`：pretrained model directory, here we set as `${project-dir}/model/`
- `model_save_dir`：the directory where the best model to be saved, here we set as `${project-dir}/saved_models/`
- `best_model_save_name`：the filename of the best model, here we set as `best_cino.pth`

#### Step 1：Model preparation
Download CINO model from [Download](https://github.com/ymcui/Chinese-Minority-PLM/blob/main/README_EN.md#Download) section, and unzip it into `${project-dir}/model/` .
The folder should contain 3 files, including `pytorch_model.bin`, `sentencepiece.bpe.model`, `config.json`.

#### Step 2：Data Preparation
Download  data from [Korean Text Classification (YNAT)](https://github.com/ymcui/Chinese-Minority-PLM/blob/main/README_EN.md#korean-text-classification-ynat) section, put them into `${data-dir}` and rename the train set and dev set as `train.txt` and `dev.txt` respectively.

#### Step 3：Run command
```shell
python ynat_finetune.py --params cino-params.json
```
`params` should be  a JSON dictionary, in this tutorial, `cino-params.json` contains all parameters  for finetuning, for example：
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
    "class_names":["Politics", "Economy", "Society", "Culture", "World", "IT/Science", "Sport"]
}
```

After running this program, you could check the log messages  and model testing results on the dev set in `${project-dir}/log/cino_ynat.log`.

