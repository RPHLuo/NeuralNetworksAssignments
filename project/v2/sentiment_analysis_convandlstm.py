import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

"""## Setup some configurational parameters"""

MAX_NB_WORDS=200000
MAX_SEQUENCE_LENGTH=50
VALIDATION_SPLIT = .2

with open("./refined_fb_data.csv", encoding = "ISO-8859-1" ) as f:
    li=f.readlines()

texts = []  # list of text samples
labels = []  # list of label ids


for row in li:
    row = row.replace('"',"").strip().split(",")
    texts.append(row[-1])
    labels.append(row[0])

print('Found %s texts.' % len(texts))

labels.count(1)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters="")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

embeddings_index = {}
f = open('./glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# data = np.array(sequences)
labels_1 = to_categorical(np.asarray(labels),num_classes=2)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_1.shape)

EMBEDDING_DIM=50
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels_1 = labels_1[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels_1[:-nb_validation_samples]
x_val = data[-nb_validation_samples:-int(nb_validation_samples/2)]
y_val = labels_1[-nb_validation_samples:-int(nb_validation_samples/2)]
x_test = data[-int(nb_validation_samples/2):]
y_test = labels_1[-int(nb_validation_samples/2):]

from keras.layers import Embedding,Input, Conv1D, MaxPooling1D, Dense, Flatten, Reshape, Dropout, LSTM, Activation
from keras.models import Model, Sequential


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
model_1 = Model(sequence_input, embedded_sequences)
model_1.summary()
model = Sequential()
model.add(model_1)
model.add(Conv1D(64, 5, activation='relu',input_shape=(None, 500)))

model.add(Dropout(0.25))
model.add(LSTM(64))
model.add(Dense(50))
model.add(Dense(2,activation="softmax"))


model.summary()

"""### Load weights"""

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# fitting the data
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=6, batch_size=1280)

score = model.evaluate(x_test,y_test, batch_size=1280)
print("Loss: "+str(score[0]))
print("Accuracy: "+str(score[1]))


model_json = model.to_json()
with open("model_tweets_new.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model_tweets_new.h5")
