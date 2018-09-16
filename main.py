import sys

import numpy as np

# PyPath linking for this project
sys.path.insert(0, '/Users/mne/Documents/ECG-NN-Classifier/ecg-nn-classifier/')

from feature_extraction.annotation_reader import AnnotationReader
from model_selection.data_shuffle import equally_distributed
from model_selection.augmentation import shift_signals
from sklearn.preprocessing import LabelEncoder
from nn import dense_ecg_net
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from validation.metrics import precision, recall


def main(**kwargs):
    print('Reading data files...')

    dims = 4
    # Creates a feature extractor for raw data on annotations
    reader = AnnotationReader()
    # Create train/test split
    X_train, X_test, y_train, y_test = equally_distributed(reader.X, reader.y, split=0.25)

    print('Train / test sets created!')

    enc = LabelEncoder()
    enc.fit(reader.y)

    print('Encoding trainings labels:')

    for idx, lbl in enumerate(enc.classes_):
        print('Training label %s, encoded as value %d' % (lbl, idx))

    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)

    # Expand 4th dimension for 2D Convolution
    # X_train = np.expand_dims(X_train, dims)
    # X_test = np.expand_dims(X_test, dims)

    # Remove additional dimension when using dense network
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    print('Compile model...')

    opt = Adam(lr=0.001)

    model = dense_ecg_net()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy', precision, recall])

    print('Model compiled!')

    batch_size = 50
    epochs = 100
    steps = len(X_train) // batch_size
    shift_gen = shift_signals(X_train, y_train, batch_size, dims)
    csv_logger = CSVLogger('%s.log' % model.name)
    plateau = ReduceLROnPlateau(min_lr=0.00001, patience=5)
    checkpoint = ModelCheckpoint(filepath='%s.hdf5' % model.name, verbose=1, save_best_only=True,
                                 monitor='val_categorical_accuracy', mode='max')

    print('Keras callbacks created...')
    print('Begin training with parameters: epochs %d, batch_size %d' % (epochs, batch_size))

    model.fit_generator(shift_gen, validation_data=(X_test, y_test), epochs=epochs, steps_per_epoch=steps, verbose=1,
                        callbacks=[csv_logger, plateau, checkpoint])


if __name__ == '__main__':
    main()
