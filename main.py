from keras.utils.np_utils import probas_to_classes
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Reshape
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Input,Merge,merge, LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence, text
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

import numpy as np
import pandas as pd
#import data_helpers_2
from Attention import Attention
from keras import backend as K
import my_callbacks

data = pd.read_csv('kaggle_data/train.csv')
test_data = pd.read_csv('kaggle_data/test.csv')
y_train = data.is_duplicate.values
n_words = 200000
tk = text.Tokenizer(nb_words=n_words)

max_len = 20
tk.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)) + list(test_data.question1.values.astype(str)) + list(test_data.question2.values.astype(str)))
x_train_a = tk.texts_to_sequences(data.question1.values.astype(str))
x_train_a = sequence.pad_sequences(x_train_a, maxlen=max_len)

x_train_b = tk.texts_to_sequences(data.question2.values.astype(str))
x_train_b = sequence.pad_sequences(x_train_b, maxlen=max_len)

x_test_a = tk.texts_to_sequences(test_data.question1.values.astype(str))
x_test_a = sequence.pad_sequences(x_test_a, maxlen=max_len)

x_test_b = tk.texts_to_sequences(test_data.question2.values.astype(str))
x_test_b = sequence.pad_sequences(x_test_b, maxlen=max_len)

test_id = test_data.test_id.values

data_dim = 300
nb_classes = 1
neuron_count = 256
dropout = 0.2
nb_filter=128

np.random.seed(7)

TEST_FILE = 'kaggle_data/test.csv'
PRED_FILE = 'kaggle_data/predictions_v2.csv'
#(x_train_a, x_train_b, y_train,x_test_a,x_test_b,id_test, n_words, sentence_maxlen) = data_helpers_2.load_data(POS_FILE, NEG_FILE, TEST_FILE)
#print("sentence_maxlen = "+str(sentence_maxlen))
#print("num words = "+str(n_words))

shared_embedding = Embedding(n_words, data_dim, input_length=max_len, dropout=dropout)
shared_conv_1 = Convolution1D(nb_filter=nb_filter, filter_length=3, border_mode='same', activation='relu')
shared_conv_2 = Convolution1D(nb_filter=nb_filter, filter_length=4, border_mode='same', activation='relu')
shared_conv_3 = Convolution1D(nb_filter=nb_filter, filter_length=5, border_mode='same', activation='relu')
shared_maxPool = MaxPooling1D(pool_length=1)
shared_lstm = LSTM(neuron_count, return_sequences=True)
shared_attention = Attention()

text_a = Input(shape=(max_len,))
text_b = Input(shape=(max_len,))

encoder_a = shared_embedding(text_a)

encoder_a = shared_conv_1(encoder_a)
encoder_a = Dropout(dropout)(encoder_a)
#encoder_a = shared_maxPool(encoder_a)
encoder_a = shared_conv_2(encoder_a)
encoder_a = Dropout(dropout)(encoder_a)
#encoder_a = shared_maxPool(encoder_a)
encoder_a = shared_conv_3(encoder_a)
encoder_a = Dropout(dropout)(encoder_a)
encoder_a_pool = shared_maxPool(encoder_a)

encoder_a_LSTM = shared_lstm(encoder_a_pool)
encoder_a_LSTM = Dropout(dropout)(encoder_a_LSTM)

encoder_a_LSTM = LSTM(neuron_count, return_sequences=True)(encoder_a_LSTM)
encoder_a_LSTM = Dropout(dropout)(encoder_a_LSTM)


encoder_b = shared_embedding(text_b)

encoder_b = shared_conv_1(encoder_b)
encoder_b = Dropout(dropout)(encoder_b)
#encoder_b = shared_maxPool(encoder_b)
encoder_b = shared_conv_2(encoder_b)
encoder_b = Dropout(dropout)(encoder_b)
#encoder_b = shared_maxPool(encoder_b)
encoder_b = shared_conv_3(encoder_b)
encoder_b = Dropout(dropout)(encoder_b)
encoder_b_pool = shared_maxPool(encoder_b)

encoder_b_LSTM = shared_lstm(encoder_b_pool)
encoder_b_LSTM = Dropout(dropout)(encoder_b_LSTM)

encoder_b_LSTM = LSTM(neuron_count, return_sequences=True)(encoder_b_LSTM)
encoder_b_LSTM = Dropout(dropout)(encoder_b_LSTM)

encoder_a_att = shared_attention([encoder_a_LSTM, encoder_b_LSTM])
encoder_b_att = shared_attention([encoder_b_LSTM, encoder_a_LSTM])

encoder_a_att = Dropout(dropout)(encoder_a_att)
encoder_b_att = Dropout(dropout)(encoder_b_att)

merged_vec = merge([encoder_a_att,encoder_b_att], mode ='concat')

mlp = Dense(neuron_count, activation='relu')(merged_vec)
#mlp = PReLU()(mlp)
mlp = Dropout(0.2)(mlp)
mlp = BatchNormalization()(mlp)

mlp = Dense(neuron_count, activation='relu')(mlp)
#mlp = PReLU()(mlp)
mlp = Dropout(0.2)(mlp)
mlp = BatchNormalization()(mlp)

mlp = Dense(neuron_count, activation='relu')(mlp)
#mlp = PReLU()(mlp)
mlp = Dropout(0.2)(mlp)
mlp = BatchNormalization()(mlp)

mlp = Dense(neuron_count, activation='relu')(mlp)
#mlp = PReLU()(mlp)
mlp = Dropout(0.2)(mlp)
mlp = BatchNormalization()(mlp)

pred = Dense(1,activation='sigmoid')(mlp)

model = Model(input = [text_a, text_b], output = pred)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

'''
saves the model weights after each epoch if the validation loss decreased
'''
#checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
#out_batch = my_callbacks.Histories(display=100)
#model.fit([x_train_a, x_train_b], y_train, nb_epoch=2,validation_split=0.1, callbacks=[out_batch], batch_size=10, verbose=0)
model.fit([x_train_a, x_train_b], y_train, nb_epoch=20,validation_split=0.1,shuffle=True, batch_size=1024, verbose=1)

#scores = model.evaluate([x_test_a, x_test_b], y_test, verbose=0)
preds_proba = model.predict([x_test_a, x_test_b,x_test_a, x_test_b,x_test_a, x_test_b],verbose=0, batch_size = 1024)
#preds = probas_to_classes(preds_proba)

fp = open(PRED_FILE, "w")
i = 0
fp.write("test_id,is_duplicate\n")
for p in preds_proba:
    fp.write(str(test_id[i])+','+str(p[0])+'\n')
    i+=1

fp.close()
