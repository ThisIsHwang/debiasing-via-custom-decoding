import time

import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup

import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import csv

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-iml-1.3b",
    load_in_8bit=True,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b", padding_side="left")

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=128,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
#model.to("cuda:0")

print_trainable_parameters(model)

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
        return {'input_ids': torch.tensor(encodings['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encodings['attention_mask'], dtype=torch.long)}

def train(model, loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone().to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(loss)

        loss.backward()
        optimizer.step()
        scheduler.step()


def fine_tune_model(data, epochs=3, batch_size=16, max_length=50, learning_rate=1e-4, warmup_steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(data, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}")
        train(model, loader, optimizer, scheduler, device)
        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        model.save_pretrained('/home/doubleyyh/bias_mitigation/model/fine_tuned_model_' + current_time)
        break
    return model

def read_tsv(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row[i] for row in reader for i in range(0, 1)]
    return data

data = read_tsv('/home/doubleyyh/bias_mitigation/dataset/preprocessed_train2.tsv')
epoch = 1
batch_size=32
learning_rate=2e-4
warmup_steps=100
fine_tuned_model = fine_tune_model(data, epochs=1, batch_size=batch_size, learning_rate=learning_rate, warmup_steps=warmup_steps)
#get current time
current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
fine_tuned_model.save_pretrained('/home/doubleyyh/bias_mitigation/model/fine_tuned_model_' + 'epoch' + '_' + str(epoch) + '_' + 'batch_size' + '_' + str(batch_size) + '_' + 'learning_rate' + '_' + str(learning_rate) + '_' + 'warmup_steps' + '_' + str(warmup_steps) + '_' + str(current_time) + "_without_instruction")