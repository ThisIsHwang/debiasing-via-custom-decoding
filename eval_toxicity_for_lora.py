
#preprocessing winobias dataset for testing bias.

# making the dataset


import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.preprocessed_wino.WinoBiasDataset import WinoBiasDataset
import evaluate
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
import logging
import logging.handlers
import csv
from nltk.tokenize import sent_tokenize
import nltk
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    log_file = "experiment_logs.csv"
    file_handler = logging.handlers.RotatingFileHandler(log_file, mode='w', maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

def generateIt(model, tokenizer, prompts, device):
    input_ids = tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = input_ids.to('cuda')
    with torch.cuda.amp.autocast():
        generated_outputs = model.generate(**input_ids, max_length=200, do_sample=False, num_beams=1)
        generated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_outputs]
    return generated_sentences


if __name__ == "__main__":
    nltk.download('punkt')

    wino_data_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/data"
    extra_gendered_words_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/extra_gendered_words.txt"
    generalized_swaps_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/generalized_swaps_fix.txt"
    dataset = WinoBiasDataset(wino_data_dir, extra_gendered_words_dir, generalized_swaps_dir)

    #make dataloader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)

    #load model


    print(f"Using {torch.cuda.device_count()} GPUs")

    peft_model_id = "/home/doubleyyh/bias_mitigation/model/fine_tuned_model_epoch_3_batch_size_16_learning_rate_0.0002_warmup_steps_100_20230528113413_without_instruction"
    config = PeftConfig.from_pretrained(peft_model_id)
    lora_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map="auto")
    lora_model = PeftModel.from_pretrained(lora_model, peft_model_id)


    device = lora_model.device

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side='left', model_max_length=512, truncation=True, return_tensors='pt')
    toxicity = evaluate.load("toxicity")

    logger.info("iteration start for loar_model: " + "facebook/opt-iml-1.3b")
    #torch.cuda.empty_cache()
    male_continuations = []
    female_continuations = []
    male_model_completions = []
    female_model_completions = []
    for male_data, female_data in tqdm(dataloader):
        with torch.cuda.amp.autocast():
            male_generated = generateIt(lora_model, tokenizer, male_data, device)
            female_generated = generateIt(lora_model, tokenizer, female_data, device)
            male_model_completions.extend(male_generated)
            female_model_completions.extend(female_generated)
            try:
                male_continuations.extend([m.replace(d, "")for d, m in zip(male_data, male_generated)])
                female_continuations.extend([m.replace(d, "") for d, m in zip(female_data, female_generated)])
            except:
                for d, m in zip(male_data, male_generated):
                    print(d, m)
                for d, m in zip(female_data, female_generated):
                    print(d, m)

    print("iteration done for lora_model: " + "facebook/opt-iml-1.3b")

    male_toxicity = toxicity.compute(predictions=male_continuations)
    female_toxicity = toxicity.compute(predictions=female_continuations)

    male_tox_dict = dict()
    female_tox_dict = dict()
    for male_text, male_score, female_text, female_score in zip(male_model_completions, male_toxicity['toxicity'], female_model_completions, female_toxicity['toxicity']):
        male_tox_dict[sent_tokenize(male_text)[0]] = male_score
        female_tox_dict[sent_tokenize(female_text)[0]] = female_score

    male_tox_dict = (dict(sorted(male_tox_dict.items(), key=lambda item: item[1], reverse=True)))

    female_tox_dict = (dict(sorted(female_tox_dict.items(), key=lambda item: item[1], reverse=True)))

    import csv

    # Define the name of the output CSV file
    output_file = "lora_model_opt_1.3b" + "man_toxicity.csv"

    # Open the file for writing
    with open(output_file, mode='w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)

        # Write the header (dictionary keys)
        # header = data.keys()
        header = ["sentence", "toxicity"]
        csv_writer.writerow(header)

        # Write the rows (dictionary values)
        for key, value in male_tox_dict.items():
            csv_writer.writerow([key, value])

    output_file = "lora_model_opt_1.3b" + "woman_toxicity.csv"

    # Open the file for writing
    with open(output_file, mode='w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)

        # Write the header (dictionary keys)
        # header = data.keys()
        header = ["sentence", "toxicity"]
        csv_writer.writerow(header)

        # Write the rows (female_tox_dict dictionary values)
        for key, value in female_tox_dict.items():
            csv_writer.writerow([key, value])


    # calculate bias score
    import numpy as np

    male_toxicity = np.array(male_toxicity["toxicity"])
    female_toxicity = np.array(female_toxicity["toxicity"])

    absolute_diff = np.abs(male_toxicity - female_toxicity)
    mean_diff = np.mean(absolute_diff)

    logger.info("toxicity absolute_diff: " + str(absolute_diff))
    logger.info("toxicity mean_diff: " + str(mean_diff))
    logger.info("man toxicity sum: " + str(np.sum(male_toxicity)))
    logger.info("woman toxicity sum: "+ str(np.sum(female_toxicity)))



