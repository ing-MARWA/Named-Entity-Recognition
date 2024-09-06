# Named-Entity-Recognition

Overview
This repository is a comprehensive project for Named Entity Recognition (NER), a core task in Natural Language Processing (NLP). NER aims to identify and classify specific entities such as people, organizations, locations, and events from unstructured text. The project demonstrates building an NER model using deep learning techniques with Keras and includes visualizations for insights into model predictions.

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
Named Entity Recognition (NER) is used to extract structured data from unstructured text. This notebook demonstrates the process of building and training an NER model using various NLP and deep learning libraries, including Keras, TensorFlow, and SpaCy.

Key entity types in the NER task:

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
Before running the project, ensure that the following dependencies are installed:

bash
Copy code
pip install numpy pandas keras spacy matplotlib seaborn
Dataset
The project utilizes the Named Entity Recognition (NER) Corpus. It contains text data labeled with various named entity types, and the model is designed to extract these entities.

Data Preprocessing
Clean Data: The data is cleaned by converting strings in the POS and Tag columns into lists, ensuring proper format for analysis.
Normalization: Tags are converted to uppercase for consistency.
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
The NER model is built using a Bidirectional LSTM network:

Embedding Layer: Converts words into dense vectors.
Bidirectional LSTM: Captures dependencies in both forward and backward directions.
TimeDistributed Dense Layer: Predicts entity tags for each word.
python
Copy code
model = Sequential()
model.add(Embedding(input_dim=Vocab_Size+1, output_dim=128, input_length=max_length))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Training
The model is trained using an early stopping mechanism to prevent overfitting, monitoring the validation loss and saving the best-performing model.

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
Plot the training and validation accuracy and loss to evaluate model performance:

python
Copy code
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
Inference
Predict named entities in a given sentence:

python
Copy code
def get_entities(sentence):
    input_seq = tokenizer.texts_to_sequences([sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding="post")
    predictions = model.predict(input_seq)
    predicted_tags = np.argmax(predictions, axis=-1)[0]
    return [(sentence.split()[i], tag_tokenizer.index_word[tag]) for i, tag in enumerate(predicted_tags) if tag != 0]
Visualization
Visualize entities in a sentence using matplotlib:

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
This repository provides a practical implementation of Named Entity Recognition using modern deep learning techniques. It showcases the full pipeline from data preprocessing to model training, evaluation, and visualization.

References
NER Dataset on Kaggle
Keras Documentation
SpaCy
