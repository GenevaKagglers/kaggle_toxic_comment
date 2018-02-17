from keras.layers import Embedding, Flatten, Dense, SpatialDropout1D, Dropout, GlobalMaxPool1D, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Input, Model, load_model
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
import defines

# Configurables
max_features = 100000
embedding_dim = 100
max_len = 50
batch_size = 32
f_neutral = 0.02
name = 'keras_test'

# Load the train data
df_train = pd.read_csv(os.path.join(defines.DATA_DIR, 'train.csv'))
df_train['comment_text'].fillna('unknown', inplace=True)

# Keep only part of the neutral comments
idx_neutral = np.where(df_train[defines.CLASS_COLS].sum(axis=1) == 0.)[0]
df_train.drop(labels=np.random.choice(idx_neutral, np.int(np.floor((1 - f_neutral) * len(idx_neutral)))), inplace=True)

# To numpy
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

# Create the model
inputs = Input(shape=(max_len,), dtype='int32')
embedding_layer = Embedding(max_features, embedding_dim, input_length=max_len)
sequence_input = Input(shape=(max_len,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequences)
x = SpatialDropout1D(0.25)(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(6, activation='sigmoid')(x)
model = Model(sequence_input, preds)
model.summary()

# Compile
callbacks = [EarlyStopping(monitor='val_auc', min_delta=0.005, patience=5, verbose=1, mode='auto'),
             ModelCheckpoint(filepath='/tmp/weights.h5', verbose=1, save_best_only=True)]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', auc])

# Fit the model
model.fit(X_train, y_train, epochs=30, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=callbacks)
model.save(os.path.join(defines.MODEL_DIR, '%s.h5' % name))

# model = load_model('/tmp/weights.h5', custom_objects={'auc': auc})

# Prepare the test set for submission
X_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# Score the model
y_pred = model.predict(X_test, batch_size=batch_size)

# Write submission
df_out = pd.DataFrame(np.concatenate((ids.reshape(-1, 1), y_pred), axis=1), columns=['id'] + defines.CLASS_COLS)
df_out.to_csv(os.path.join(defines.SUB_DIR, '{}.csv'.format(name)), index=False)
