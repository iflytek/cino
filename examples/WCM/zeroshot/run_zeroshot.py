import os
import random
import json, time
import logging
import argparse
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.nn.modules import padding
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, AdamW, get_cosine_schedule_with_warmup

logging.basicConfig(filename='log/cino_zs.log', filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("CINO Logger")


def load(json_file):

    if json_file.endswith(".json"):
        with open(json_file, 'r') as f:
            return json.load(f)
    else:
        return json.loads(json_file)


class CINO_Model(nn.Module):

    def __init__(self, cino_path, class_num):
        
        super(CINO_Model, self).__init__()
        self.config = XLMRobertaConfig.from_pretrained(cino_path)
        self.cino = XLMRobertaModel.from_pretrained(cino_path)
        for param in self.cino.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.config.hidden_size, class_num)

    def forward(self, input_ids, attention_mask):

        output = self.cino(input_ids, attention_mask)[1]
        logit = self.fc(output)
        return logit

class CINO_ZS_Configer():

    def __init__(self, params_dict: dict):
        
        super().__init__()
        
        self.batch_size = params_dict['batch_size']
        self.max_len = params_dict['max_len']
        self.model_pretrain_dir = params_dict['model_pretrain_dir']
        self.model_finetune_params = params_dict['model_finetune_params']
        self.class_names = params_dict['class_names']
        self.data_dir = params_dict['data_dir']

class CINO_ZeroShoter():

    def __init__(self, config: CINO_ZS_Configer):

        super().__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.config.model_pretrain_dir)

    def dataset(self, data_path):

        input_ids, attention_masks, labels = [], [], []
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                text, label = line.strip().split('\t')
                encode_dict = self.tokenizer.encode_plus(text=text, \
                                                        max_length=self.config.max_len, \
                                                        padding='max_length', \
                                                        truncation=True)
                input_ids.append(encode_dict['input_ids'])
                attention_masks.append(encode_dict['attention_mask'])
                labels.append(int(label))
        
        return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)
    
    def data_loader(self, input_ids, attention_masks, labels):

        data = TensorDataset(input_ids, attention_masks, labels)
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)

        return loader

    def predict(self, model, data_loader):

        model.eval()
        test_pred, test_true = [], []
        with torch.no_grad():
            for idx, (ids, att, y) in enumerate(data_loader):
                y_pred = model(ids.to(self.device), att.to(self.device))
                y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
                test_pred.extend(y_pred)

                y_temp = y.squeeze().cpu().numpy().tolist()
                test_true.extend([y_temp] if type(y_temp) == int else y_temp)
            return test_true, test_pred


    def zeroshot(self):

        model = CINO_Model(self.config.model_pretrain_dir, class_num=len(self.config.class_names)).to(self.device)
        model.load_state_dict(torch.load(self.config.model_finetune_params))

        for data_file in glob(f"{self.config.data_dir}*.txt"):
            lan = data_file.split('/')[-1][:-4]
            logger.info("\tCurrent Language : " + lan)
            test_loader = self.data_loader(*self.dataset(data_file))

            test_true, test_pred = self.predict(model, test_loader)
            logger.info('\n\n' + classification_report(test_true, test_pred,
                                                target_names=self.config.class_names,
                                                digits=6))

def main():

    parser = argparse.ArgumentParser(description="CINO-Argparser")
    parser.add_argument("--params", default={}, help="JSON dict of model hyperparameters.")   
    args = parser.parse_args()

    config = CINO_ZS_Configer(load(args.params))

    trainer = CINO_ZeroShoter(config)
    trainer.zeroshot()

if __name__ == "__main__":
    main()
