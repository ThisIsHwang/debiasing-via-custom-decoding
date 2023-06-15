# Text Debiasing via Custom Decoding

This repository contains the implementation of our custom decoding method to mitigate toxicity in text generation, using the Large Language Models.

## Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Usage](#usage)
5. [Results](#results)
6. [License](#license)
7. [Contact](#contact)

## Installation
To run this project, you need to have Python installed on your machine. The project mainly depends on PyTorch, Hugging Face's Transformers, and Datasets libraries. You can install these packages using `pip`:
```bash
pip install torch transformers datasets
```
You will also need to install the following packages for evaluation:
```bash
pip install rouge bert_score nltk
```

## Project Motivation
This project is aimed at addressing the toxicity issue in the output of language models. We proposed a custom decoding method that checks the toxicity level of the generated output and regenerates the output if it exceeds a certain threshold.

## File Descriptions
- `eval_summarization.py`: This is the main Python script where the generation and evaluate summarization.
- `eval_toxicity.py`: This script is used to load the toxicity model for the evaluation of generated text.

# Toxicity measurement

By using 'evaluate' library, we can compute toxicity value. The toxicity measurement is based on "roberta-hate-speech-dynabench-r4" model.

```python
import evaluate
toxicity = evaluate.load("toxicity")
```

```python
results = toxicity.compute(predictions=input_texts)
```

<br/>
<br/>

# Generating text using customed decoder

First, start a virtual environment inside your directory, and activate the virtual environment.
```python
python -m venv .env
```
```python
source .env/bin/activate
``` 
<br/>

Install 'evaluate' and 'transformers' library.
```python
pip install evaluate, transformers
```
<br/>
<br/>

**_From now on, below codes are included in 'eval_toxicity_for_decoding.py', you may execute single file or follow the below codes._**

<br/>
<br/>

Instantiate an evaluation module "toxicity".
```python
import evaluate
toxicity = evaluate.load("toxicity")
```

<br/>

Then, load pretrained model and tokenizer from LLaMA.
```python
from transformers import LlamaTokenizer, LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("/home/doubleyyh/gpt4all-13b-snoozy", device_map="auto", load_in_8bit=True)
```
```python
tokenizer = LlamaTokenizer.from_pretrained("/home/doubleyyh/gpt4all-13b-snoozy", model_max_length=30, padding_side='left')
```

<br/>

Set several hyperparameters' values. You can customize below values.
```python
hyp_toxic_threshold = 0.4
hyp_prob = 1
hyp_wordnum = 3
hyp_penalty_init = 1
hyp_penalty_rate = -0.05
hyp_dict = {'toxicity_threshold': hyp_toxic_threshold, 'regenerate_probability': hyp_prob,
            'num_word': hyp_wordnum, 'initial_penalty': hyp_penalty_init, 'penalty_rate': hyp_penalty_rate}
```

<br/>

Set 'tot_len' value (desired length of generated text), and make an input prompt text.

```python
tot_len = 30
text = input("prompt: ")
```
<br/>


Then, import the customed decoder file and generate the text.
```python
from debiasing-via-custom-decoding import eval_toxicity_for_decoding
generated_text_1 = eval_toxicity_for_decoding.custom_generate(model, tokenizer, toxicity, text, tot_len, hyp_dict)
```

```python
generated_text_1 = tokenizer.batch_decode(generated_text_1.long(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print('customed generation : ', generated_text_1)
```


<br/>

## Usage
You can run the main script using Python:
```bash
python eval_summarization.py
python eval_toxicity.py
```
You can modify hyperparameters in the `eval_toxicity.py` file, such as the toxicity threshold and the number of words to regenerate.

## Results
The system generates a CSV file with the results of the evaluation. It contains the articles, reference summaries, system summaries, and the ROUGE, BLEU, and BERT scores for each summary.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions, feel free to open an issue or pull request.
doubleyyh@kaist.ac.kr


# Citation
```python
@inproceedings{vidgen2021lftw,
  title={Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection},
  author={Bertie Vidgen and Tristan Thrush and Zeerak Waseem and Douwe Kiela},
  booktitle={ACL},
  year={2021}
}
```

```python
@article{gehman2020realtoxicityprompts,
  title={Realtoxicityprompts: Evaluating neural toxic degeneration in language models},
  author={Gehman, Samuel and Gururangan, Suchin and Sap, Maarten and Choi, Yejin and Smith, Noah A},
  journal={arXiv preprint arXiv:2009.11462},
  year={2020}
}
```

```python
@inproceedings{Wolf_Transformers_State-of-the-Art_Natural_2020,
author = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Perric and Ma, Clara and Jernite, Yacine and Plu, Julien and Xu, Canwen and Le Scao, Teven and Gugger, Sylvain and Drame, Mariama and Lhoest, Quentin and Rush, Alexander M.},
month = oct,
pages = {38--45},
publisher = {Association for Computational Linguistics},
title = {{Transformers: State-of-the-Art Natural Language Processing}},
url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6},
year = {2020}
}
```



