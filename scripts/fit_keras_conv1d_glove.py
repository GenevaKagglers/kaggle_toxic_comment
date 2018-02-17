from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, Dropout, GlobalMaxPool1D, BatchNormalization, \
    Conv1D, MaxPooling1D
from keras.models import Sequential, Input, Model, load_model
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from utils import *
from textutils import StemTokenizer
import defines

# Configurables
max_features = 100000
embedding_dim = 300
max_len = 40
batch_size = 32
f_neutral = 0.02
name = 'keras_glove_test'

embeddings_index = {}
f = open(os.path.join(defines.GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = ' '.join(values[:-embedding_dim])
    try:
        coefs = np.asarray(values[-embedding_dim:], dtype='float32')
    except Exception as e:
        print(word)
        print(line)
        import sys
        sys.exit()
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Load the train data
df_train = pd.read_csv(os.path.join(defines.DATA_DIR, 'train.csv'))
df_train['comment_text'].fillna('unknown', inplace=True)
idx_neutral = np.where(df_train[defines.CLASS_COLS].sum(axis=1) == 0.)[0]
df_train.drop(labels=np.random.choice(idx_neutral, np.int(np.floor((1 - f_neutral) * len(idx_neutral)))), inplace=True)
X_train = df_train[defines.INPUT_COL].as_matrix()
y_train = df_train[defines.CLASS_COLS].as_matrix()

# Load test set
df_test = pd.read_csv(os.path.join(defines.DATA_DIR, 'test.csv'))
df_test['comment_text'].fillna('unknown', inplace=True)
ids = df_test['id'].as_matrix()
X_test = df_test[defines.INPUT_COL].as_matrix()

# Create and fit the tokenizer
tokenizer = text.Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower=False, split=" ", char_level=False)
tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))

# Prepare the sequences for the training
X_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(X_train, maxlen=max_len, truncating='pre', padding='post')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Initialize the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Build the model
inputs = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)
sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequences)
x = SpatialDropout1D(0.25)(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(6, activation='sigmoid')(x)
model = Model(sequence_input, preds)
model.summary()

# Compile
callbacks = [EarlyStopping(monitor='val_auc', min_delta=0.005, patience=5, verbose=1, mode='max'),
             ModelCheckpoint(filepath='/tmp/weights.h5', verbose=1, save_best_only=True)]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', auc])

# Fit the model
model.fit(X_train, y_train, epochs=20, batch_size=batch_size, shuffle=True, validation_split=0.1, callbacks=callbacks)
model.save(os.path.join(defines.MODEL_DIR, '%s.h5' % name))

# model = load_model('/tmp/weights.h5', custom_objects={'auc': auc})

# Prepare the test set for submission
X_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# SCore the model
y_pred = model.predict(X_test, batch_size=batch_size)

# Write submission
df_out = pd.DataFrame(np.concatenate((ids.reshape(-1, 1), y_pred), axis=1), columns=['id'] + defines.CLASS_COLS)
df_out.to_csv(os.path.join(defines.SUB_DIR, '{}.csv'.format(name)), index=False)
