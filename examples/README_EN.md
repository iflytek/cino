## Usage

This folder contains examples for finetuning our pretrained model CINO and testing it on zero-shot tasks.

### Requirements
- numpy : 1.21.2
- python : 3.7.10
- pytorch : 1.7.1
- scikit-learn : 0.24.2
- transformers : 3.1.0

### Finetune

In this tutorial, we will finetune `CINO-large` with Chinese datasets in WCM.
- `project-dir`：working directory
- `data-dir`：data directory, here we set as `${project-dir}/data/`
- `model_pretrain_dir`：pretrained model directory, here we set as `${project-dir}/model/`
- `model_save_dir`：the directory where the best model to be saved, here we set as `${project-dir}/saved_models/`
- `best_model_save_name`：the filename of the best model, here we set as `best_cino.pth`

#### Step 1：Model preparation
Download CINO model from [Download](#模型下载) section, and unzip it into `${project-dir}/model/` .
The folder should contain 3 files, including `pytorch_model.bin`, `sentencepiece.bpe.model`, `config.json`.

#### Step 2：Data Preparation
Download Chinese data from [Wiki-Chinese-Minority（WCM）](#少数民族语言分类数据集) section, put them into `${data-dir}` and rename them as `train.txt`, `dev.txt` and `txt`.

#### Step 3：Run command
```shell
python run_finetune.py --params cino-params.json
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
    "class_names":["Arts", "Geography", "History", "Nature", "Natural Science", "Personage", "Technology", "Education", "Economy", "Health"] 
}
```

After running this program, you could check the log messages  and model testing results in `${project-dir}/log/cino-ft.log`.


### Zero-Shot

In this tutorial, we will test the zero-shot performance of our finetuned CINO model.

- `project-dir`：working directory
- `data-dir`：data directory, here we set as `${project-dir}/data/`
- `model_pretrain_dir`：pretrained model directory, here we set as  `${project-dir}/model/`
- `model_finetune_params`：the path of the best model parameters file saved in finetuning stage, here we set as `${project-dir}/model/best_cino.pth`

#### Step 1：Model preparation
Download CINO model from [Download](#模型下载) section, and unzip it into `${project-dir}/model/`.
The folder should contain 3 files, including `pytorch_model.bin`, `sentencepiece.bpe.model`, `config.json`.

Then, put the best model parameters file saved in finetuning stage into `${project-dir}/model/`.

#### Step 2：Data Preparation
Download minority language data  from [Wiki-Chinese-Minority（WCM）](#少数民族语言分类数据集) section including `bo.txt`, `kk.txt`, `ko.txt`, `mn.txt`, `ug.txt` and `yue.txt` and put them into `${data-dir}`.

#### Step 3：Run command
```shell
python run_zeroshot.py --params cino-params.json
```
`params` should be  a JSON dictionary, in this tutorial, `cino-params.json` contains all paramters  for finetuning, for example:
```json
{
    "batch_size":16,
    "max_len":512,
    "model_pretrain_dir":"model/",
    "model_finetune_params":"model/best_cino.pth",
    "data_dir":"data/",
    "class_names":["Arts", "Geography", "History", "Nature", "Natural Science", "Personage", "Technology", "Education", "Economy", "Health"]
}
```

After running this program, you could check the log messages  and zero-shot results in `${project-dir}/log/cino-zs.log`.

