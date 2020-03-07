import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
import keras.backend as K
from data_loader import load_data

def binary_crossentropy(y_true, y_pred):
    return K.mean(-10 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred))


def create_model_mask_prediction():
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(3, 3), strides=2,
                     activation='relu',
                     input_shape=(128, 128, 1)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(1, (2, 2), activation='sigmoid'))

    model.compile(loss=binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()
    return model



def evaluate_mask_model(model, data, masks):
    # Consider values > 0.5 as 1
    out = model.predict(data) > 0.5

    # Create arrays for accuracy, precision and recall
    acc = []
    precision = []
    recall = []

    for i in range(len(out)):
        # Fill with predicted masks
        predicted_vector = out[i].reshape(-1, )
        # Fill with given masks
        target_vector = masks[i].reshape(-1, )

        # Calculate accuracy
        cur_acc = np.sum(predicted_vector == target_vector) / len(target_vector)

        # TP = np.sum(target_vector*predicted_vector)
        # FP = np.sum((predicted_vector * -1) * (target_vector - 1))
        # FN = np.sum(((predicted_vector - 1) * (target_vector * -1)))

        TP = np.sum(np.logical_and(target_vector == predicted_vector, predicted_vector == True))
        FP = np.sum(np.logical_and(target_vector != predicted_vector, predicted_vector == True))
        # TN = np.sum(np.logical_and(target_vector == predicted_vector, predicted_vector == False))
        FN = np.sum(np.logical_and(target_vector != predicted_vector, predicted_vector == False))

        # Calculate Precision
        cur_precision = TP / (TP + FP)

        # Calculate Recall
        cur_recall = TP / (TP + FN)

        acc.append(cur_acc)
        precision.append(cur_precision)
        recall.append(cur_recall)

    # Removing nan values from arrays
    precision = np.asarray(precision)
    precision = precision[np.logical_not(np.isnan(precision))]
    recall = np.asarray(recall)
    recall = recall[np.logical_not(np.isnan(recall))]

    # Calculate f-measure
    f1 = 2 * ((np.mean(precision) * np.mean(precision)) / (np.mean(precision) + np.mean(precision)))

    # Results
    print("Accuracy is: ", np.mean(acc), "+-", np.std(acc) )
    print("Precision is: ", np.mean(precision), "+-", np.std(precision))
    print("Recall is: ", np.mean(recall), "+-",  np.std(recall))
    print("f1:", np.mean(f1))


def create_model_pneumonia_fc():
    model_2 = Sequential()

    model_2.add(Dense(256, activation='relu', input_shape=(169, )))

    model_2.add(Dense(1, activation='sigmoid'))

    model_2.compile(loss=keras.losses.binary_crossentropy,
                    optimizer=keras.optimizers.Adam(lr=0.001),
                    metrics=['accuracy'])

    model_2.summary()
    return model_2
