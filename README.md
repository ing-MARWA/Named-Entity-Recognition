# Named-Entity-Recognition

Named Entity Recognition (NER) Project
Overview
This project demonstrates the implementation of Named Entity Recognition (NER), a critical task in Natural Language Processing (NLP). The goal of NER is to identify entities such as people, organizations, locations, dates, and more from unstructured text data. The project utilizes a deep learning-based approach using Keras and includes visualizations to gain insights into model predictions.

Table of Contents
Introduction
Project Setup
Dataset
Data Preprocessing
Model Architecture
Training
Evaluation
Inference
Visualization
Conclusion
References
Introduction
Named Entity Recognition (NER) is a powerful tool in NLP, enabling the extraction of structured data from unstructured text. This project focuses on building a deep learning model for NER using a Bidirectional LSTM network.

The key entity types recognized in this project are:

geo: Geographical Entity
org: Organization
per: Person
gpe: Geopolitical Entity
tim: Time indicator
art: Artifact
eve: Event
nat: Natural Phenomenon
Project Setup
Dependencies
Ensure the following dependencies are installed before running the project:

bash
Copy code
pip install numpy pandas keras spacy matplotlib seaborn
Dataset
The dataset used in this project is the Named Entity Recognition (NER) Corpus, which contains labeled text data for various entity types.

Data Preprocessing
The following steps are applied during data preprocessing:

Data Cleaning: Converts string representations in POS and Tag columns into lists for easier manipulation.
Normalization: Ensures that tags are standardized by converting them to uppercase.
python
Copy code
def preprocess_data(data):
    for i in range(len(data)):
        pos = ast.literal_eval(data['POS'][i])
        tags = ast.literal_eval(data['Tag'][i])
        data['POS'][i] = [str(word) for word in pos]
        data['Tag'][i] = [str(word.upper()) for word in tags]
    return data
Model Architecture
The model architecture is built using Bidirectional LSTM layers for better context understanding. The main components are:

Embedding Layer: Converts words into dense vector representations.
Bidirectional LSTM: Captures the context in both forward and backward directions.
TimeDistributed Dense Layer: Predicts entity tags for each word in the sentence.
python
Copy code
model = Sequential()
model.add(Embedding(input_dim=Vocab_Size+1, output_dim=128, input_length=max_length))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Training
The model is trained using early stopping to prevent overfitting. The model is optimized using the Adam optimizer, and the training process is monitored by validation loss.

python
Copy code
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_inputs_final, train_targets_final,
    validation_data=(test_inputs_final, test_targets_final),
    batch_size=64,
    epochs=15,
    callbacks=[early_stopping]
)
Evaluation
After training, the model's performance is evaluated by plotting the training and validation accuracy and loss.

python
Copy code
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
Inference
To predict entities in a new sentence, the following function is used. It processes the input text and returns the predicted named entities.

python
Copy code
def get_entities(sentence):
    input_seq = tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post")
    predictions = model.predict(input_seq)
    predicted_tags = np.argmax(predictions, axis=-1)[0]
    return [(sentence.split()[i], tag_tokenizer.index_word[tag]) for i, tag in enumerate(predicted_tags) if tag != 0]
Visualization
The predicted entities in a sentence are visualized using matplotlib, with each entity highlighted according to its type.

python
Copy code
def visualize_entities(sentence, entities):
    words = sentence.split()
    fig, ax = plt.subplots(figsize=(len(words) * 0.6, 1))
    for entity, entity_type in entities:
        start_index = words.index(entity)
        ax.axhspan(0, 1, xmin=start_index / len(words), xmax=(start_index + 1) / len(words), facecolor=get_color(entity_type), alpha=0.5)
    plt.show()
Conclusion
This project demonstrates how to build, train, and evaluate an NER model using deep learning techniques. It also highlights how to visualize model predictions to gain better insights into the NER task.

References
NER Dataset on Kaggle
Keras Documentation
SpaCy
