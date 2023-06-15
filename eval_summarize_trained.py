import os
from rouge import Rouge
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

lora_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-iml-1.3b",
)


from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
peft_model_id = "/home/doubleyyh/bias_mitigation/model/fine_tuned_model_t"
config = PeftConfig.from_pretrained(peft_model_id)
lora_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True)
lora_model = PeftModel.from_pretrained(lora_model, peft_model_id)
lora_model.to(device)

# model = OPTForCausalLM.from_pretrained("facebook/opt-iml-1.3b")
# model.to(device)

#smoothing_function for BLEU
smoothing_function = SmoothingFunction().method1

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b")

# Load cnn_dailymail_v002 dataset
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")

# Split the dataset into train, validation, and test sets
train_set, val_set, test_set = cnn_dailymail["train"], cnn_dailymail["validation"], cnn_dailymail["test"]

# Initialize Rouge object
rouge = Rouge()


def generate_summary(text, model=lora_model, tokenizer=tokenizer, device=device):
    # Replace this function with your own summarization model
    # For exmple, you can use a pre-trained T5 or BART model
    inputs = tokenizer.encode("summarize: " + text + "\nanswer: ", return_tensors="pt", max_length=1000,
                              truncation=True)
    inputs = inputs.to(device)
    # Generate the summary
    summary_ids = model.generate(input_ids=inputs, max_length=1500, do_sample=False, num_beams=1)
    # Decode the summary
    text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return text


def calculate_rouge_f1(system_summary, reference_summary):
    scores = rouge.get_scores(system_summary, reference_summary, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

def calculate_blue_score(system_summary, reference_summary):
    reference_tokens = reference_summary.split()
    candidate_tokens = system_summary.split()
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
    return bleu_score

import csv


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Article', 'Reference Summary', 'System Summary', 'ROUGE1', "ROUGE2", "ROUGEL", "BLUE"])

        for result in results:
            writer.writerow([result['article'], result['reference_summary'], result['system_summary'], result['rouge_1'], result['rouge_2'], result['rouge_l'], result["blue"]])


# Evaluate on the test set and store results
results = []
# showing the progress of the evaluation
import tqdm

for example in tqdm.tqdm(test_set):
    reference_summary = example["highlights"]
    system_summary = generate_summary(text=example["article"], model=lora_model)
    reference_tokens = reference_summary.split()
    candidate_tokens = system_summary.split()

    rouge1, rouge2, rouge_l = calculate_rouge_f1(system_summary, reference_summary)
    blue = calculate_blue_score(system_summary, reference_summary)

    result = {
        'article': example["article"],
        'reference_summary': reference_summary,
        'system_summary': system_summary,
        'rouge_1': rouge1,
        'rouge_2': rouge2,
        'rouge_l': rouge_l,
        "blue": blue
    }
    results.append(result)

# Save results to a CSV file
save_results_to_csv(results, 'eval_summarize_trained_results.csv')

# Compute the average ROUGE F-1 score
average_rouge_f1 = sum([result['rouge_1'] for result in results]) / len(results)
average_rouge_f2 = sum([result['rouge_2'] for result in results]) / len(results)
average_rouge_fl = sum([result['rouge_l'] for result in results]) / len(results)
average_blue = sum([result['blue'] for result in results]) / len(results)

#save the scores with current time
import datetime
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
print("Current Time =", current_time)

#save the scores
with open('eval_trained_summarize_scores.txt', 'a') as f:
    print("Current Time =", current_time, file=f)
    print("Average ROUGE F-1 Score:", average_rouge_f1, file=f)
    print("Average ROUGE F-2 Score:", average_rouge_f2, file=f)
    print("Average ROUGE F-L Score:", average_rouge_fl, file=f)
    print("Average BLEU Score:", average_blue, file=f)

# Print the average ROUGE F-1 score
print("Current Time =", current_time)
print("Average ROUGE F-1 Score:", average_rouge_f1)
print("Average ROUGE F-2 Score:", average_rouge_f2)
print("Average ROUGE F-L Score:", average_rouge_fl)
print("Average BLEU Score:", average_blue)

