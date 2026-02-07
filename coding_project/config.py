import os
import pandas as pd

# Paths
IMAGE_DIR = os.path.join('images_training_rev1', 'images_training_rev1')
CSV_PATH = os.path.join('training_solutions_rev1', 'training_solutions_rev1.csv')

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

# Automatically count classes from CSV
df = pd.read_csv(CSV_PATH)

# Drop the first column (GalaxyID) to get just the class labels
class_columns = df.columns[1:]
NUM_CLASSES = len(class_columns)

print(f"Detected {NUM_CLASSES} classes: {list(class_columns)}")
