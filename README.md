# Mamba-BERT: Exploring Falcon-Mamba-7B for NLP Tasks

## Overview
This repository contains code and configurations for experimenting with Falcon-Mamba-7B, Llama 3.2-1B, and BERT on downstream NLP tasks, including:
- **Text Classification**: IMDB (binary sentiment), CoLA (grammatical acceptability), and CORONA (multi-class sentiment).
- **Text Summarization**: CNN/DailyMail dataset.

The project explores the effectiveness of Falcon-Mamba-7B, a non-attention-based language model, compared to traditional transformer-based models like BERT and Llama 3.2-1B. 

## Features
- Fine-tuning scripts for Falcon-Mamba-7B.
- Preprocessing scripts for datasets.
- Evaluation metrics for classification (accuracy) and summarization (ROUGE).
- Baseline comparison with BERT and Llama 3.2-1B.

## Datasets
1. **IMDB**: Binary sentiment classification.
2. **CoLA**: Grammatical acceptability classification.
3. **CORONA**: Multi-class sentiment classification.
4. **CNN/DailyMail**: Abstractive summarization dataset.

Download datasets from:
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [CoLA Dataset](https://nyu-mll.github.io/CoLA/)
- [CORONA Dataset](https://www.kaggle.com/datasets)
- [CNN/DailyMail Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

Place the datasets in the `./` directory.


## Results
| Dataset       | Model               | Accuracy (%) | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) |
|---------------|---------------------|--------------|-------------|-------------|-------------|
| IMDB          | Falcon-Mamba-7B    | 91           | -           | -           | -           | 
| CoLA          | Falcon-Mamba-7B    | 60           | -           | -           | -           |         
| CORONA        | Falcon-Mamba-7B    | 83.2         | -           | -           | -           |         
| CNN/DailyMail | Falcon-Mamba-7B    | -            | 36.2        | 18.1        | 24.7        |      

## Notes
- LoRA fine-tuning is used for parameter efficiency.
- Falcon-Mamba-7B performs better on classification tasks compared to BERT and Llama 3.2-1B.
