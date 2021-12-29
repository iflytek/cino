import json, time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import padding
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, AdamW, get_cosine_schedule_with_warmup

logging.basicConfig(filename='log/cino_ynat.log', filemode="w",
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

class CINO_FT_Configer():

    def __init__(self, params_dict: dict):
        
        super().__init__()
        
        self.learning_rate = params_dict['learning_rate']
        self.epoch = params_dict['epoch']
        self.gradient_acc = params_dict['gradient_acc']
        self.batch_size = params_dict['batch_size']
        self.max_len = params_dict['max_len']
        self.model_save_dir = params_dict['model_save_dir']
        self.best_model_save_name = params_dict['best_model_save_name']
        self.warmup_rate = params_dict['warmup_rate']
        self.class_names = params_dict['class_names']
        self.weight_decay = params_dict['weight_decay']
        self.data_dir = params_dict['data_dir']
        self.model_pretrain_dir = params_dict['model_pretrain_dir']

class CINO_Trainer():

    def __init__(self, config: CINO_FT_Configer):

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
                test_true.extend(y.squeeze().cpu().numpy().tolist())
            return test_true, test_pred


    def train(self, model, train_loader, valid_loader, optimizer, schedule):

        best_f1 = 0.0
        criterion = nn.CrossEntropyLoss()
        for i in range(self.config.epoch):
            start_time = time.time()
            train_loss_sum = 0.0
            model.train()
            logger.info(f"—————————————————————— Epoch {i+1} ——————————————————————")
            
            for idx, (ids, att, y) in enumerate(train_loader):

                ids, att, y = ids.to(self.device), att.to(self.device), y.to(self.device)
                y_pred = model(ids, att)
                loss = criterion(y_pred, y)
                loss /= self.config.gradient_acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                train_loss_sum += loss.item()

                if (idx+1) % (len(train_loader) // 10) == 0:
                    logger.info("Epoch {:02d} | Step {:03d}/{:03d} | Loss {:.4f} | Time {:.2f}".format( \
                                i+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time()-start_time))
            
            model.eval()
            dev_true, dev_pred = self.predict(model, valid_loader)
            weighted_f1 = f1_score(dev_true, dev_pred, average='weighted')

            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                torch.save(model.state_dict(), self.config.model_save_dir+self.config.best_model_save_name)
            
            logger.info("Current dev weighted f1 is {:.4f}, best f1 is {:.4f}".format(weighted_f1, best_f1))
            logger.info("Time costed : {}s \n".format(round(time.time() - start_time, 3)))
    

    def run_finetune(self):

        train_loader = self.data_loader(*self.dataset(f'{self.config.data_dir}train.txt'))
        dev_loader = self.data_loader(*self.dataset(f'{self.config.data_dir}dev.txt'))

        model = CINO_Model(self.config.model_pretrain_dir, class_num=len(self.config.class_names)).to(self.device)
        

        total_steps = (len(train_loader) // self.config.batch_size + 1) * self.config.epoch
        optimizer = AdamW(params=model.parameters(), 
                        lr=self.config.learning_rate, 
                        weight_decay=self.config.weight_decay)      
        schedule = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=self.config.warmup_rate*total_steps,
                                                    num_training_steps=total_steps)
        self.train(model, train_loader, dev_loader, optimizer, schedule)

        model.load_state_dict(torch.load(self.config.model_save_dir+self.config.best_model_save_name))
        dev_true, dev_pred = self.predict(model, dev_loader)

        logger.info('\n\n' + classification_report(dev_true, dev_pred,
                                                target_names=self.config.class_names,
                                                digits=6))

def main():

    parser = argparse.ArgumentParser(description="CINO-Argparser")
    parser.add_argument("--params", default={}, help="JSON dict of model hyperparameters.")   
    args = parser.parse_args()

    config = CINO_FT_Configer(load(args.params))

    trainer = CINO_Trainer(config)
    trainer.run_finetune()

if __name__ == "__main__":
    main()

