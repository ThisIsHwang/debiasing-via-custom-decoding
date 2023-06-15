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


