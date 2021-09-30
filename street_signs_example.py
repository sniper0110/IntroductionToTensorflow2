import os
import glob

from PIL.Image import FASTOCTREE
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from my_utils import split_data, order_test_set, create_generators

from deeplearning_models import streesigns_model
import tensorflow as tf


if __name__=="__main__":


    path_to_train = "D:\\Datasets\\GTSRB\\raw_downloaded_dataset\\GTSRB-GermanTrafficSignRecognitionBenchmark\\training_data\\train"
    path_to_val = "D:\\Datasets\\GTSRB\\raw_downloaded_dataset\\GTSRB-GermanTrafficSignRecognitionBenchmark\\training_data\\val"
    path_to_test = "D:\\Datasets\\GTSRB\\raw_downloaded_dataset\\GTSRB-GermanTrafficSignRecognitionBenchmark\\Test"
    batch_size = 64
    epochs = 15
    lr=0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN=False
    TEST=True

    if TRAIN:
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = streesigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
                )

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set : ")
        model.evaluate(test_generator)
