import time
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_datasets
from training import train_model
from config import EPOCHS, LR

# Dataset fractions for scaling
fractions = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

# Hyperparameters for sensitivity analysis
optimizers = ["nadam", "sgd"]
learning_rates = [1e-4, 1e-3]

# Store results
results = []

def get_optimizer(name, lr):
    if name.lower() == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    elif name.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError("Unsupported optimizer")

# Define the EarlyStopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True,
    verbose=1 # To see when early stopping is triggered
)

def evaluate_and_report(model, val_ds, phase_name, run_details=""):
    y_true = []
    y_pred = []
    for images, labels in val_ds.unbatch(): # Unbatch to get individual samples
        y_true.append(labels.numpy())
        prediction = model.predict(tf.expand_dims(images, axis=0), verbose=0)
        y_pred.append(np.argmax(prediction))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n--- Classification Report for {phase_name} {run_details}---")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {phase_name} {run_details}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Calculate overall accuracy, precision, recall, f1 from the report
    overall_accuracy = report['accuracy']
    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']

    return overall_accuracy, weighted_precision, weighted_recall, weighted_f1


# === 1. Sensitivity Analysis ===
print("=== Sensitivity Analysis ===")
for opt in optimizers:
    for lr in learning_rates:
        print(f"\nOptimizer: {opt}, Learning rate: {lr}")
        train_ds, val_ds = load_datasets(subset_fraction=0.05)
        optimizer = get_optimizer(opt, lr)

        model, _, sensitivity_training_time_sec = train_model(optimizer, train_ds, val_ds, EPOCHS, callbacks=[early_stopping_callback])

        acc, prec, rec, f1 = evaluate_and_report(model, val_ds, "Sensitivity Analysis", f"Opt:{opt}, LR:{lr}")
        print(f"Validation Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

# === 2. Training Time vs Dataset Size ===
print("\n=== Training Time vs Dataset Size ===")
best_acc = 0.0
best_model = None

for frac in fractions:
    print(f"\nDataset fraction: {frac}")
    train_ds, val_ds = load_datasets(subset_fraction=frac)
    optimizer = get_optimizer("nadam", LR) # Using Nadam with default LR for this section

    model, _, training_time_sec = train_model(optimizer, train_ds, val_ds, EPOCHS, callbacks=[early_stopping_callback])

    acc, prec, rec, f1 = evaluate_and_report(model, val_ds, "Dataset Size", f"Fraction:{frac}")
    print(f"Training time: {training_time_sec:.2f} sec, Validation Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    results.append({
        "fraction": frac,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "training_time_sec": training_time_sec
    })

    if acc > best_acc:
        best_acc = acc
        best_model = model
        model.save("best_model.keras")
        print("Saved new best model.")

# === 3. Inference Time ===
print("\n=== Inference Time ===")
if best_model is None:
    print("No best model saved from training time vs dataset size. Loading the last trained model for inference.")
    # If the loop above didn't save a best model (e.g., due to errors or if only one fraction was run)
    # this might need adjustment to ensure 'best_model' is always available if needed.
    # For this subtask, assuming best_model will be set by the previous loop.
    # Alternatively, load the best_model.keras if it exists from a previous run or manual save.
    best_model = tf.keras.models.load_model('best_model.keras') # Load the saved best model

_, val_ds_full = load_datasets(subset_fraction=1.0) # Use the full validation set for inference testing
images, _ = next(iter(val_ds_full)) # Get a batch of images from the full validation dataset
repeats = 10
times = []

for _ in range(repeats):
    start = time.time()
    best_model.predict(images, verbose=0)
    end = time.time()
    times.append(end - start)

avg_inference_time = sum(times) / len(times)
print(f"Average inference time per batch: {avg_inference_time:.4f} sec")

# === 4. Save final trained parameters to CSV ===
print("\n=== Saving final trained parameters to CSV ===")
all_weights = []
for i, layer in enumerate(best_model.layers):
    weights = layer.get_weights()
    for j, w in enumerate(weights):
        all_weights.append({
            "layer_index": i,
            "layer_name": layer.name,
            "weight_index": j,
            "weight_shape": str(w.shape),
            "weights_flat": w.flatten().tolist()  # flatten to 1D
        })

# Convert to DataFrame
df_weights = pd.DataFrame(all_weights)
df_weights.to_csv("best_model_weights.csv", index=False)
print("Saved trained parameters to 'best_model_weights.csv'.")

# === 5. Summary Table ===
print("\n=== Summary of Results ===")
for r in results:
    print(f"Fraction: {r['fraction']*100:.0f}%, "
          f"Accuracy: {r['accuracy']:.4f}, "
          f"Precision: {r['precision']:.4f}, "
          f"Recall: {r['recall']:.4f}, "
          f"F1-score: {r['f1_score']:.4f}, "
          f"Training time: {r['training_time_sec']:.2f} sec")
