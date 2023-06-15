import os
from rouge import Rouge
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import random
import evaluate

smoothing_function = SmoothingFunction().method1

lora_model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-1.3b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b", padding_side="left", truncation=True, model_max_length=2048, return_tensors="pt")
# Load cnn_dailymail_v002 dataset
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
device = lora_model.device

hyp_toxic_threshold = 0.4  # if the word's toxicity is over this threshold, the word is considered as toxic
hyp_prob = 1  # probability to regenerate if toxic word captured
hyp_wordnum = 3  # num of words to regenerate if toxic word captured
hyp_penalty_init = 1  # initial penalty value (penalty : if same toxic word regenerated, give penalty)
hyp_penalty_rate = -0.05  # decrease rate of penalty according to its' position

hyp_dict = {'toxicity_threshold': hyp_toxic_threshold, 'regenerate_probability': hyp_prob,
            'num_word': hyp_wordnum, 'initial_penalty': hyp_penalty_init, 'penalty_rate': hyp_penalty_rate}

sfm = torch.nn.Softmax()


def custom_generate(model, tokenizer, toxicity, text, tot_len, hyp_dict):
    generating_text = text
    cnt = 0  # count of generating process >> if loop doesn't stops, manually stop by checking cnt
    pos = 0  # word generating position >> adjusting penalty according to pos
    penalty_dict = dict()  # ['toxic word's id' : lately-captured-position]

    result_ids = tokenizer.encode(generating_text, return_tensors='pt').to(model.device)
    tot_len = len(result_ids[0]) + 128
    #print(tot_len)
    generated_words = torch.tensor([[]]).to(model.device)
    while True:
        if len(result_ids[
                   0]) + len(generated_words[0]) >= tot_len:  # if generated text satisfies intended length or iteration repeats too much
            break
        outputs = model(torch.concat((result_ids.to(model.device), generated_words.to(model.device)), dim=-1).long().to(model.device))

        #give me a logit that is the probability of next word
        next_token_logits = outputs[0][:, -1, :]

        softmax_prob = sfm(next_token_logits.float())
        #softmax_prob = sfm(next_token_logits.float())

        # 각 단어들 확률에 penalty 적용
        for key in list(penalty_dict.keys()):
            penalty = hyp_dict['initial_penalty'] + hyp_dict['penalty_rate'] * abs(
                pos - penalty_dict[key])  # 현재 위치와 toxic-capture된 위치 사이 거리에 따라 감소하는 penalty
            # print("applied penalty : ", penalty)  # pos에 따라 감소하여 적용되는 penalty check
            if penalty < 0:  # capture된 거리~현재 거리 일정 이상 멀어지면 penalty 0 부여
                penalty = 0
            if softmax_prob[0][key].item() > penalty:
                softmax_prob[0][key] -= penalty
            else:
                softmax_prob[0][key] = 0

        pred_id = torch.argmax(softmax_prob).item()
        # append to result_ids
        generated_words = torch.cat([generated_words.to(model.device), torch.tensor([[pred_id]]).to(model.device)], dim=-1).to(model.device)
        # decode result_ids
        sentence_for_toxicity_check = tokenizer.decode(generated_words[0].long(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if toxicity.compute(predictions=[sentence_for_toxicity_check])['toxicity'][0] < hyp_dict[
            'toxicity_threshold']:
            pos += 1  # word-adding position (0 : position of first-generated-word)
        else:  # toxic word captured
            regen = random.choices([0, 1],
                                   weights=[1 - hyp_dict['regenerate_probability'], hyp_dict['regenerate_probability']])
            if regen[0]:  # regenerate by prob. of regeneration. If word_num differs, change position as pos -= (word_num)
                ## add the toxic word in penalty dictionary. if already exists, change to latest pos value
                ## word_num-1 만큼 단어 빼주고 pos도 word_num-1 만큼 더 빼준다 (-1 : 아직 새로운 단어 추가 안했으므로)
                if pos > hyp_dict['num_word']:
                    new_right_ind = -hyp_dict['num_word']
                    # I want to check whether the number of the result_ids is bigger than initial_prompt_len
                    # if it is bigger, I want to delete the words from the initial_prompt_len to the new_right_ind
                    if len(generated_words[0]) > hyp_dict['num_word']:
                        for i, key in enumerate(generated_words[0][len(generated_words[0]) - hyp_dict['num_word']:]):
                            penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                        generated_words = generated_words[0][:new_right_ind].unsqueeze(0).to(model.device)
                        pos -= hyp_dict['num_word']
                        #penalty_dict[pred_id] = pos - 1
                    else: # if regenerating hurts the original prompt : only get rid of one 'toxic word'
                        for i, key in enumerate(generated_words[0]):
                            penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                        generated_words = torch.tensor([[]]).to(model.device)
                        pos = 0
                else:
                    for i, key in enumerate(generated_words[0]):
                        penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                    #generated_words = torch.tensor([[]]).to(model.device)
                    pos = 0
                    cnt += 1
                    continue
        cnt += 1
    return torch.concat((result_ids.to(model.device), generated_words.to(model.device)), dim=-1).to(model.device)

# Split the dataset into train, validation, and test sets
train_set, val_set, test_set = cnn_dailymail["train"], cnn_dailymail["validation"], cnn_dailymail["test"]

# Initialize Rouge object
rouge = Rouge()

toxicity = evaluate.load("toxicity")

def generate_summary(text, model=lora_model, tokenizer=tokenizer, device=device):
    # Replace this function with your own summarization model
    # For example, you can use a pre-trained T5 or BART model
    inputs = "summarize: " + text + "\nanswer: "
    # Generate the summary
    with torch.cuda.amp.autocast():

        summary_ids = custom_generate(model, tokenizer, toxicity, inputs, 1500, hyp_dict)  # customed
        text = tokenizer.decode(*summary_ids.long(), skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
        # Decode the summary
        #print(text)
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

hyp_penalty_init = [0.3, 0.5, 0.7, 1]
for h in hyp_penalty_init:
    hyp_dict['initial_penalty'] = h
    # Evaluate on the test set and store results
    results = []
    # showing the progress of the evaluation
    import tqdm
    #dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    cnt = 0
    for example in tqdm.tqdm(test_set):
        if cnt > 100:
            break
        cnt +=1
        reference_summary = example["highlights"]
        print(len(example["article"]))
        if len(example["article"]) > 5000:

            continue
        system_summary = generate_summary(example["article"])
        try:
            system_summary = system_summary.split("answer:")[1]
            system_summary = system_summary.split("I'm not sure if this is a good idea")[0]
            print(system_summary)
        except:
            print(system_summary)
        P, R, F1 = score([system_summary], [reference_summary], lang='en', device=lora_model.device)
        rouge1, rouge2, rouge_l = calculate_rouge_f1(system_summary, reference_summary)
        blue = calculate_blue_score(system_summary, reference_summary)

        result = {
            'article': example["article"],
            'reference_summary': reference_summary,
            'system_summary': system_summary,
            'rouge_1': rouge1,
            'rouge_2': rouge2,
            'rouge_l': rouge_l,
            "blue": blue,
            "bert_score": float(F1)
        }
        results.append(result)

    # Save results to a CSV file
    save_results_to_csv(results, 'eval_peft_summarize_results.csv')

    # Compute the average ROUGE F-1 score
    average_rouge_f1 = sum([result['rouge_1'] for result in results]) / len(results)
    average_rouge_f2 = sum([result['rouge_2'] for result in results]) / len(results)
    average_rouge_fl = sum([result['rouge_l'] for result in results]) / len(results)
    average_blue = sum([result['blue'] for result in results]) / len(results)
    average_bert_score = sum([result['bert_score'] for result in results]) / len(results)

    #save the scores with current time
    import datetime
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    print("Current Time =", current_time)

    #save the scores
    with open('eval_peft_summarize_scores.txt', 'a') as f:
        print("Current Time =", current_time, file=f)
        print("Average ROUGE F-1 Score:", average_rouge_f1, file=f)
        print("Average ROUGE F-2 Score:", average_rouge_f2, file=f)
        print("Average ROUGE F-L Score:", average_rouge_fl, file=f)
        print("Average BLEU Score:", average_blue, file=f)
        print("Average BERT Score:", average_bert_score, file=f)

    # Print the average ROUGE F-1 score
    print("Current Time =", current_time)
    print("Average ROUGE F-1 Score:", average_rouge_f1)
    print("Average ROUGE F-2 Score:", average_rouge_f2)
    print("Average ROUGE F-L Score:", average_rouge_fl)
    print("Average BLEU Score:", average_blue)
    print("Average BERT Score:", average_bert_score)

