from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaTokenizer, LlamaForCausalLM, OPTForCausalLM
import torch as pt
import random
import evaluate
import numpy as np

##parameters
hyp_toxic_threshold = 0.4
hyp_prob = 1
hyp_wordnum = 3
hyp_penalty_init = 1
hyp_penalty_rate = -0.05
hyp_dict = {'toxicity_threshold': hyp_toxic_threshold, 'regenerate_probability': hyp_prob,
            'num_word': hyp_wordnum, 'initial_penalty': hyp_penalty_init, 'penalty_rate': hyp_penalty_rate}

sfm = pt.nn.Softmax()

def custom_generate(model, tokenizer, toxicity, text, tot_len, hyp_dict):
    
    generating_text = text
    cnt = 0  # count of process step
    pos = 0  # word generating position
    penalty_dict = dict()  # ['toxic word's id' : lately-captured-position]

    initial_prompt_token = tokenizer.tokenize(text)
    initial_prompt_len = len(initial_prompt_token)
    result_ids = tokenizer.encode(generating_text, return_tensors='pt').to(model.device)
    generated_words = pt.tensor([[]]).to(model.device)
    
    while True:
        if len(result_ids[
                   0]) + len(generated_words[0]) >= tot_len:
            break

        ## logit that is the probability of next word
        outputs = model(pt.concat((result_ids, generated_words), dim=-1).long().to(model.device))
        next_token_logits = outputs[0][:, -1, :]
        softmax_prob = sfm(next_token_logits.float())

        ## penalty application
        for key in list(penalty_dict.keys()):
            penalty = hyp_dict['initial_penalty'] + hyp_dict['penalty_rate'] * abs(
                pos - penalty_dict[key])
            if penalty < 0:
                penalty = 0
            if softmax_prob[0][key].item() > penalty:
                softmax_prob[0][key] -= penalty
            else:
                softmax_prob[0][key] = 0

        ## token-attaching
        pred_id = pt.argmax(softmax_prob).item()
        generated_words = pt.cat([generated_words, pt.tensor([[pred_id]]).to(model.device)], dim=-1).to(model.device)
        pred_word = tokenizer.decode(pred_id)
        sentence_for_toxicity_check = tokenizer.decode(generated_words[0].long(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(sentence_for_toxicity_check)

        ## toxic/nontoxic case branches
        if toxicity.compute(predictions=[sentence_for_toxicity_check])['toxicity'][0] < hyp_dict[
            'toxicity_threshold']:
            pos += 1 
        else:
            regen = random.choices([0, 1],
                                   weights=[1 - hyp_dict['regenerate_probability'], hyp_dict['regenerate_probability']])
            if regen[0]:
                if pos > hyp_dict['num_word']:
                    new_right_ind = -hyp_dict['num_word']
                    if len(generated_words[0]) > hyp_dict['num_word']:
                        for i, key in enumerate(generated_words[0][len(generated_words[0]) - hyp_dict['num_word']:]):
                            penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                        generated_words = generated_words[0][:new_right_ind].unsqueeze(0).to(model.device)
                        pos -= hyp_dict['num_word']
                    else:
                        for i, key in enumerate(generated_words[0]):
                            penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                        generated_words = pt.tensor([[]]).to(model.device)
                        pos = 0
                else:
                    for i, key in enumerate(generated_words[0]):
                        penalty_dict[int(key)] = pos - hyp_dict['num_word'] + i + 1
                    generated_words = pt.tensor([[]]).to(model.device)
                    pos = 0
                    cnt += 1
                    continue
        cnt += 1

    return pt.concat((result_ids, generated_words.to(model.device)), dim=-1).to(model.device)

tot_len = 30

toxicity = evaluate.load("toxicity")

tokenizer = LlamaTokenizer.from_pretrained("/home/doubleyyh/gpt4all-13b-snoozy", model_max_length=30, padding_side='left')
model = LlamaForCausalLM.from_pretrained("/home/doubleyyh/gpt4all-13b-snoozy", device_map="auto", load_in_8bit=True)

while True:
    text = input("prompt: ")
    print("-" * 100)
    generated_text_1 = custom_generate(model, tokenizer, toxicity, text, tot_len, hyp_dict)
    generated_text_1 = tokenizer.batch_decode(generated_text_1.long(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print('customed generation : ', generated_text_1)
    print("-" * 100)