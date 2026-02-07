import time
import tensorflow as tf
from model import build_model

def train_model(optimizer, train_ds, val_ds, epochs, verbose=0, callbacks=None):
    model = build_model()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose, callbacks=callbacks)
    end = time.time()

    training_time_sec = end - start
    print(f"Training time for this run: {training_time_sec:.2f} seconds")

    return model, history, training_time_sec
