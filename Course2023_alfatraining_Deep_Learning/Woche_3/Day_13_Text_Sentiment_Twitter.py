# Python, using Anaconda environment
# Week 1, Day 13

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

# loading data from a CSV file (CSV2)
# FileName = r"C:\Users\alfa\PycharmProjects\python_deeplearning\Woche_3\sentiment_Shrivastava.csv"
FileName = r"data\sentiment_Shrivastava.csv"
data_df = pd.read_csv(FileName, delimiter=";", encoding="ISO-8859-1")
print(data_df, data_df.size,  type(data_df))
print("\n")

is_all_data_samples = False
num_samples = 25500  # 1000 samples as default

if is_all_data_samples:
    print("Using all data samples \n")
    data_concat = data_df
else:
    half_samples = int(num_samples / 2)
    num_samples = 2 * half_samples
    print(f"Using {num_samples} samples \n")
    # choose n samples, e.g. 1000 samples
    # hint:  https://sparkbyexamples.com/pandas/get-first-n-rows-of-pandas/
    data_df_first = data_df.iloc[:half_samples]
    print(data_df_first, data_df_first.size, type(data_df_first))
    data_df_last = data_df.iloc[-half_samples:]
    print(data_df_last, data_df_last.size, type(data_df_last))
    print("\n")

    # hint:  https://pandas.pydata.org/docs/user_guide/merging.html
    frames = [data_df_first, data_df_last]
    data_concat = pd.concat(frames)
print(data_concat, data_concat.size, type(data_concat))
print("\n")


# select output (sentiment)
y = data_concat.loc[:, "polarity of tweet"].to_numpy()
# print(type(y))
# print(min(y), max(y))
y = y/4.0
# print(type(y[0]))
# print(y.shape)

# select input
X = data_concat.loc[:, "text of the tweet"].to_numpy()
# print(type(X), X.shape)
# print(X[0], type(X[0]))

# train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters='"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ', char_level=False, oov_token=None)

tokenizer.fit_on_texts(X_train)
# print(type(tokenizer.word_index))
# print(tokenizer.word_index)

X_train_encoded = tokenizer.texts_to_sequences(X_train)
# print(type(X_train_encoded), X_train_encoded)
X_test_encoded = tokenizer.texts_to_sequences(X_test)
# print(X_test_encoded)

# set how many maximum words per tweets
max_length = 20  # 8 or 16 or 20
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_encoded,
                                                               truncating='post',
                                                               maxlen=max_length,
                                                               padding='post')
# print(type(X_train_padded), X_train_padded)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_encoded,
                                                              truncating='post',
                                                              maxlen=max_length,
                                                              padding='post')
# print(type(X_test_padded), X_test_padded)
print(len(tokenizer.word_index))

# Für die Embedding-Schicht brauchen wir die Anzahl der Wörter aus dem Tokenizer
# Für das Padding kommt noch die 0 als weiteres wort hinzu, also +1
vocab_size = len(tokenizer.word_index) + 1

word_vector_dimension = 3  # 3-dimension word-vector

# Model
text_model = tf.keras.models.Sequential()
text_model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dimension,
                                         input_length=max_length,
                                         mask_zero=True))
# text_model.add(tf.keras.layers.Flatten())
text_model.add(tf.keras.layers.LSTM(100, return_sequences=False))
text_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

text_model.summary()

text_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# model fitting/learning
text_model.fit(X_train_padded, y_train, epochs=50, verbose=True, batch_size=5)

# validation
y_pred = text_model.predict(X_test_padded)
# print(y_test)
# print(y_pred)
print("\n")

# metrics
train_loss, train_accuracy = text_model.evaluate(X_train_padded, y_train, verbose=0)
print(f"training accuracy:  {train_accuracy * 100} %  \n")

test_loss, test_accuracy = text_model.evaluate(X_test_padded, y_test, verbose=0)
print(f"test accuracy:  {test_accuracy * 100} %  \n")


# Word Vektoren grafisch darstellen, if it is in 3 dimension
do_i_want_picture = False
if word_vector_dimension == 3 and do_i_want_picture:
    X_wordVect = text_model.layers[0](np.arange(vocab_size))

    # 1. Dimension der Wort-Vektoren auf X-Achse,
    # 2. Dimension auf y-Achse, 3. auf die Z-Achse abbilden
    ax = plt.figure().add_subplot(projection='3d')

    ax.view_init(10, 20)
    ax.scatter3D(X_wordVect[:, 0], X_wordVect[:, 1], X_wordVect[:, 2], color='red')
    for i in range(vocab_size - 1):
        ax.text(X_wordVect[i + 1, 0], X_wordVect[i + 1, 1], X_wordVect[i + 1, 2],
                list(tokenizer.word_index.keys())[i])
    plt.show()




# end of file
