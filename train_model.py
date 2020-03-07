import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from exps.data_loader import load_data

train_data, train_masks, train_labels, test_data, test_labels, test_masks = load_data()

import numpy as np

# Reshape data to be compatible with keras
train_data = train_data.reshape((-1, train_data.shape[1], train_data.shape[2], 1))
test_data = test_data.reshape((-1, test_data.shape[1], test_data.shape[2], 1))
train_labels = keras.utils.to_categorical(train_labels, 2)
test_labels = keras.utils.to_categorical(test_labels, 2)


def create_model():
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(3, 3), strides=2,
                     activation='relu',
                     input_shape=(128, 128, 1)))

    model.add(MaxPooling2D(pool_size=(3,3)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (2, 2),strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    #model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    return model


def evaluate_model(model):
    class_weight = {0: 1.,
                   1: 1.2,
                    }
    model.fit(train_data, train_labels,
              batch_size=128,
              epochs=15,
              verbose=1,
              validation_data=(test_data, test_labels), class_weight=class_weight)
    train_score = model.evaluate(train_data, train_labels, verbose=0)
    test_score = model.evaluate(test_data, test_labels, verbose=0)
    return train_score, test_score


model = create_model()
print(evaluate_model(model))
