# Named-Entity-Recognition

```
# Named Entity Recognition (NER) with Deep Learning

This repository contains a Python notebook that demonstrates how to build a Named Entity Recognition (NER) model using deep learning techniques. The model is trained on a NER dataset and can identify entities such as people, organizations, locations, and more within text.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [License](#license)

## Introduction

Named Entity Recognition (NER) is a natural language processing (NLP) task that involves identifying and classifying named entities in text. These entities can be names of people, organizations, locations, dates, times, and more. NER is a crucial component in various NLP applications, including information extraction, question answering, and text summarization.

This project implements a Bidirectional LSTM (Long Short-Term Memory) neural network for NER. Bidirectional LSTMs are well-suited for sequence labeling tasks like NER as they can capture the context of a word from both directions (past and future).

## Dataset

The model is trained on the "Named Entity Recognition (NER) Corpus" dataset available on Kaggle:

- **Dataset Source:** [https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus)
- **Entities:** The dataset includes the following entity types:
    - **geo:** Geographical Entity
    - **org:** Organization
    - **per:** Person
    - **gpe:** Geopolitical Entity
    - **tim:** Time indicator
    - **art:** Artifact
    - **eve:** Event
    - **nat:** Natural Phenomenon

## Requirements

To run the notebook, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- TensorFlow or Keras
- spacy
- matplotlib
- seaborn

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn tensorflow spacy matplotlib seaborn
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Download the dataset from the provided link.

3. Update the dataset path in the notebook.

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook ner-notebook.ipynb
   ```

## Model Architecture

The NER model is built using a Bidirectional LSTM architecture:

1. **Embedding Layer:** Maps words to their corresponding word embeddings.
2. **Bidirectional LSTM Layer:** Processes the input sequence in both directions to capture contextual information.
3. **Dropout Layer:** Added for regularization to prevent overfitting.
4. **TimeDistributed Dense Layer:** Predicts the NER tag for each word in the sequence.

## Evaluation

The model is evaluated on a held-out test set using metrics such as accuracy, precision, recall, and F1-score. The notebook provides a detailed analysis of the model's performance.

## Visualization

The notebook includes visualizations to help understand the model's predictions, including:

- **Entity Visualization:** Highlights named entities in a given sentence.
- **Probability Distribution:** Shows the predicted probabilities for each NER tag.

## License

This project is licensed under the MIT License. See the LICENSE file for details. 
``` 
