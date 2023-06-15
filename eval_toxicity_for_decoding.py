
#preprocessing winobias dataset for testing bias.

# making the dataset

import torch as pt
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset.preprocessed_wino.WinoBiasDataset import WinoBiasDataset
import evaluate
from transformers import AutoTokenizer, OPTForCausalLM
import logging
import logging.handlers
import csv
from nltk.tokenize import sent_tokenize
import nltk

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
    input_ids = input_ids["input_ids"].to(device)
    generated_outputs = model.module.generate(input_ids, max_length=50, do_sample=False, num_beams=1, early_stopping=True)
    generated_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_outputs]
    return generated_sentences

hyp_toxic_threshold = 0.4  # if the word's toxicity is over this threshold, the word is considered as toxic
hyp_prob = 1  # probability to regenerate if toxic word captured
hyp_wordnum = 3  # num of words to regenerate if toxic word captured
hyp_penalty_init = 0.3  # initial penalty value (penalty : if same toxic word regenerated, give penalty)
hyp_penalty_rate = -0.05  # decrease rate of penalty according to its' position

hyp_dict = {'toxicity_threshold': hyp_toxic_threshold, 'regenerate_probability': hyp_prob,
            'num_word': hyp_wordnum, 'initial_penalty': hyp_penalty_init, 'penalty_rate': hyp_penalty_rate}

sfm = pt.nn.Softmax()

def custom_generate(model, tokenizer, toxicity, text, tot_len, hyp_dict):
    generating_text = text[0]
    cnt = 0  # count of generating process >> if loop doesn't stops, manually stop by checking cnt
    pos = 0  # word generating position >> adjusting penalty according to pos
    penalty_dict = dict()  # ['toxic word's id' : lately-captured-position]

    result_ids = tokenizer.encode(generating_text, return_tensors='pt').to(model.module.device)
    generated_words = pt.tensor([[]]).to(model.module.device)
    while True:
        if len(result_ids[
                   0]) + len(generated_words[0]) >= tot_len:  # if generated text satisfies intended length or iteration repeats too much
            break
        #input_ids = tokenizer.encode(generating_text, return_tensors='pt')

        #print(len(input_ids[0]))  # check the length of generating sentence
        # give me input_ids logits
        #type casting to torch.Long() is needed because of the error : expected dtype torch.int64 but got dtype torch.int32

        outputs = model(pt.concat((result_ids.to(model.module.device), generated_words.to(model.module.device)), dim=-1).long().to(model.module.device))

        #give me a logit that is the probability of next word
        next_token_logits = outputs[0][:, -1, :]

        softmax_prob = sfm(next_token_logits.float())
        #softmax_prob = sfm(next_token_logits.float())

        # 각 단어들 확률에 penalty 적용
        for key in list(penalty_dict.keys()):
            # print(tokenizer.decode(key))
            # print(softmax_prob[0][key])
            penalty = hyp_dict['initial_penalty'] + hyp_dict['penalty_rate'] * abs(
                pos - penalty_dict[key])  # 현재 위치와 toxic-capture된 위치 사이 거리에 따라 감소하는 penalty
            # print("applied penalty : ", penalty)  # pos에 따라 감소하여 적용되는 penalty check
            if penalty < 0:  # capture된 거리~현재 거리 일정 이상 멀어지면 penalty 0 부여
                penalty = 0
            if softmax_prob[0][key].item() > penalty:
                softmax_prob[0][key] -= penalty
            else:
                softmax_prob[0][key] = 0

        pred_id = pt.argmax(softmax_prob).item()
        # append to result_ids
        generated_words = pt.cat([generated_words.to(model.module.device), pt.tensor([[pred_id]]).to(model.module.device)], dim=-1).to(model.module.device)
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
                        generated_words = generated_words[0][:new_right_ind].unsqueeze(0).to(model.module.device)
                        pos -= hyp_dict['num_word']
                        #penalty_dict[pred_id] = pos - 1
                    else: # if regenerating hurts the original prompt : only get rid of one 'toxic word'
                        for i, key in enumerate(generated_words[0]):
                            penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                        generated_words = pt.tensor([[]]).to(model.module.device)
                        pos = 0
                else:
                    for i, key in enumerate(generated_words[0]):
                        penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                    generated_words = pt.tensor([[]]).to(model.module.device)
                    pos = 0
                    cnt += 1
                    continue
        cnt += 1
    return torch.concat((result_ids.to(model.module.device), generated_words.to(model.module.device)), dim=-1).to(model.module.device)

hyp_penalty_init = [0.3, 0.5, 0.7, 1]
for h in hyp_penalty_init:
    hyp_dict['initial_penalty'] = h
    if __name__ == "__main__":
        nltk.download('punkt')

        wino_data_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/data"
        extra_gendered_words_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/extra_gendered_words.txt"
        generalized_swaps_dir = "/home/doubleyyh/bias_mitigation/corefBias/WinoBias/wino/generalized_swaps_fix.txt"
        dataset = WinoBiasDataset(wino_data_dir, extra_gendered_words_dir, generalized_swaps_dir)

        #make dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        #load model


        print(f"Using {torch.cuda.device_count()} GPUs")

        model = OPTForCausalLM.from_pretrained("facebook/opt-iml-1.3b", device_map="auto", load_in_8bit=True)
        model = torch.nn.DataParallel(model)
        device = model.module.device
        #model.to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b", model_max_length=512, padding_side='left')
        toxicity = evaluate.load("toxicity")

        logger.info("iteration start for decoding model: " + "facebook/opt-iml-1.3b")
        #torch.cuda.empty_cache()
        male_continuations = []
        female_continuations = []
        male_model_completions = []
        female_model_completions = []
        cnt = 0
        for male_data, female_data in tqdm(dataloader):
            if cnt == 100:
                break
            cnt += 1
            with torch.no_grad():
                male_generated = custom_generate(model, tokenizer, toxicity, male_data, 50, hyp_dict)  # customed
                male_generated = tokenizer.batch_decode(male_generated.long(), skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=True)[0]  # not customed
                #male_generated = generateIt(model, tokenizer, male_data, device)
                female_generated = custom_generate(model, tokenizer, toxicity, female_data, 50, hyp_dict)  # customed#generateIt(model, tokenizer, female_data, device)
                female_generated = tokenizer.batch_decode(female_generated.long(), skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=True)[0]  # not customed
                male_model_completions.append(male_generated)
                female_model_completions.append(female_generated)
                male_continuations.append(sent_tokenize(male_generated.replace(male_data[0], ""))[0])
                female_continuations.append(sent_tokenize(female_generated.replace(female_data[0], ""))[0])

        print("iteration done for decoding model: " + "facebook/opt-iml-1.3b")

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
        output_file = "decoding_man_toxicity.csv"

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

        output_file = "decoding_woman_toxicity.csv"

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




