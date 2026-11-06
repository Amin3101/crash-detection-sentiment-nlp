import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (accuracy_score,f1_score,mean_absolute_error,classification_report,confusion_matrix)



df1 = pd.read_csv("train_data.csv")
df2 = pd.read_csv("test_data.csv")


X = df1['reviewText'].fillna('')
y = df1['overall'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
MAX_LEN = 320
MAX_WORDS = 30000

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token='<OOV>'
)

tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_test = df2['reviewText'].fillna('').astype(str)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

X_val_pad = pad_sequences(
    X_val_seq,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LEN,
    padding='post',
    truncating='post'
)
y_train_model = y_train.astype('int32') - 1
y_val_model = y_val.astype('int32') - 1

class_weights_model = {
    int(k) - 1: v for k, v in class_weights.items()
}
model = Sequential()

model.add(Embedding(
    input_dim=MAX_WORDS,
    output_dim=128,
    mask_zero=True
))

model.add(Bidirectional(LSTM(64, dropout=0.2)))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.00001
)
history = model.fit(
    X_train_pad,
    y_train_model,
    validation_data=(X_val_pad, y_val_model),
    epochs=10,
    batch_size=128,
    class_weight=class_weights_model,
    callbacks=[early_stop, reduce_lr]
)
y_pred_probs = model.predict(X_val_pad, batch_size=128)

y_pred = np.argmax(y_pred_probs, axis=1)

print('Accuracy:', accuracy_score(y_val_model, y_pred))

print('Macro F1:', f1_score(
    y_val_model, y_pred, average='macro'
))
print('MAE:', mean_absolute_error(
    y_val_model + 1, y_pred + 1
))

print('\nClassification Report:')
print(classification_report(
    y_val_model,
    y_pred,
    labels=[0, 1, 2, 3, 4],
    target_names=['1', '2', '3', '4', '5'],
    zero_division=0
))
