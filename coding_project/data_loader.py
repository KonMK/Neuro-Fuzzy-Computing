import tensorflow as tf
import pandas as pd
import numpy as np
import os
from functools import partial
from config import IMG_SIZE, BATCH_SIZE

# IMG_SIZE, BATCH_SIZE, PADDING are globally defined in previous cells

# Re-define constants to ensure they are available within the cell
# In a real notebook flow, these would be accessible from previous cells
CSV_PATH = os.path.join('training_solutions_rev1', 'training_solutions_rev1.csv')
IMAGE_DIR = os.path.join('images_training_rev1', 'images_training_rev1')

def load_datasets(validation_split=0.2, subset_fraction=1.0):
    # 1. Load CSV
    df = pd.read_csv(CSV_PATH)

    # 2. Clean IDs
    image_ids = df.iloc[:, 0].astype(str).str.strip().values  # strip whitespace

    # 3. Labels: pick class with max probability
    labels_soft = df.iloc[:, 1:].values
    labels = np.argmax(labels_soft, axis=1)

    # 4. Build a map of lowercase filenames in the folder
    existing_files = {f.lower(): f for f in os.listdir(IMAGE_DIR)}

    # 5. Build filepaths safely
    filepaths = []
    valid_labels = []
    missing_files = []

    for img_id, label in zip(image_ids, labels):
        filename = f"{img_id}.jpg"
        lookup_name = filename.lower()
        if lookup_name in existing_files:
            filepaths.append(os.path.join(IMAGE_DIR, existing_files[lookup_name]))
            valid_labels.append(label)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"Warning: {len(missing_files)} files listed in CSV not found in folder.")
        print("First 10 missing:", missing_files[:10])

    if not filepaths:
        raise RuntimeError("No valid image files found. Check your IMAGE_DIR and CSV_PATH.")

    # Apply subset fraction
    if subset_fraction < 1.0:
        subset_len = max(1, int(len(filepaths) * subset_fraction))
        filepaths = filepaths[:subset_len]
        valid_labels = valid_labels[:subset_len]

    # Build TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, valid_labels))

    def load_image(path, label, img_size_val):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, img_size_val, img_size_val)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment_image(image, label, img_size_val):
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random rotation by 0, 90, 180, or 270 degrees
        if tf.random.uniform(()) > 0.5:
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Random zoom-in effect (crop a smaller portion and resize back to original IMG_SIZE)
        if tf.random.uniform(()) > 0.5:
            # Scale factor for cropping: 0.6 to 1.0 means we crop 60% to 100% of the image size.
            # Cropping a smaller percentage and resizing back simulates a zoom-in.
            scale_factor = tf.random.uniform([], minval=0.6, maxval=1.0)
            original_size = tf.cast(tf.shape(image)[0], tf.float32)
            cropped_size = tf.cast(original_size * scale_factor, tf.int32)

            # Ensure cropped_size is at least 1 pixel to avoid errors
            cropped_size = tf.maximum(1, cropped_size)

            image = tf.image.random_crop(image, size=[cropped_size, cropped_size, 3])
            image = tf.image.resize(image, [img_size_val, img_size_val])

        return image, label

    dataset = dataset.map(partial(load_image, img_size_val=IMG_SIZE), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000, seed=42)

    # Split into training and validation
    val_size = int(len(filepaths) * validation_split)
    val_ds = dataset.take(val_size)
    train_ds = dataset.skip(val_size)

    # Apply augmentation ONLY to the training dataset
    train_ds = train_ds.map(partial(augment_image, img_size_val=IMG_SIZE), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
