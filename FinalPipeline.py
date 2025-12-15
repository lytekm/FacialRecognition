"""
final_mlp_pipeline.py

End-to-end pipeline for UMIST face recognition using:
- Data loading & resizing
- Stratified train/val/test split
- Standardization
- Autoencoder for nonlinear dimensionality reduction
- MLP classifier on latent features
- Visualization of training curves
- Sample test images with true vs predicted labels
"""

import numpy as np
import pandas as pd
from skimage.transform import resize
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns


import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam

MAT_PATH = "umist_cropped.mat"
TARGET_SHAPE = (92, 112)
RANDOM_STATE = 42
NUM_PEOPLE = 20

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else np.arange(cm.shape[1]),
        yticklabels=class_names if class_names is not None else np.arange(cm.shape[0]),
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()



def load_umist_flat(mat_path: str, target_shape=(92, 112)):
    data = loadmat(mat_path)
    facedat = data["facedat"]

    all_images = []
    all_labels = []

    num_people = facedat.shape[1]

    for person_idx in range(num_people):
        person_images = facedat[0, person_idx]
        H_orig, W_orig, N_images = person_images.shape

        for img_idx in range(N_images):
            img = person_images[:, :, img_idx].astype("float32")

            # Resize to common shape
            img_resized = resize(img, target_shape, anti_aliasing=True)

            all_images.append(img_resized.flatten())
            all_labels.append(person_idx)

    X_flat = np.array(all_images)
    y = np.array(all_labels)
    H, W = target_shape

    return X_flat, y, H, W


def stratified_splits(X, y, test_ratio=0.30, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_autoencoder(input_dim: int, code_dim: int = 50):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(200, activation='relu')(input_layer)
    code = Dense(code_dim, activation='relu')(encoded)
    decoded = Dense(200, activation='relu')(code)
    output = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output)
    encoder = Model(inputs=input_layer, outputs=code)

    autoencoder.compile(optimizer=Adam(1e-3), loss="mse")
    return autoencoder, encoder


def plot_history(history, title_prefix="Model"):
    # Loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{title_prefix} – Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy (if present)
    if "accuracy" in history.history:
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["accuracy"], label="Train Acc")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.title(f"{title_prefix} – Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def build_mlp_classifier(input_dim: int, num_classes: int = NUM_PEOPLE):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def show_sample_predictions(X_test_flat, y_test, y_pred, H, W, n_samples=8):
    idxs = np.random.choice(len(X_test_flat), size=min(n_samples, len(X_test_flat)), replace=False)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs, start=1):
        img_flat = X_test_flat[idx]
        img = img_flat.reshape(H, W)
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        plt.subplot(2, 4, i)
        plt.imshow(img, cmap="gray")
        color = "green" if true_label == pred_label else "red"
        plt.title(f"T:{true_label}  P:{pred_label}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Load & preprocess
    print("Loading UMIST data and resizing...")
    X_flat, y, H, W = load_umist_flat(MAT_PATH, TARGET_SHAPE)
    print(f"X shape (flat): {X_flat.shape}, y shape: {y.shape}")
    print(f"Image size: {H} x {W}")

    # Stratified train/val/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_splits(
        X_flat, y, test_ratio=0.30, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Standardization
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_val, X_test
    )

    # Autoencoder
    input_dim = X_train_scaled.shape[1]
    code_dim = 50  # latent dimension

    print("\nBuilding autoencoder...")
    autoencoder, encoder = build_autoencoder(input_dim, code_dim=code_dim)

    print("Training autoencoder...")
    ae_history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=30,
        batch_size=64,
        verbose=1,
    )

    # Plot autoencoder training curves
    plot_history(ae_history, title_prefix="Autoencoder")

    # Encode data into latent space
    print("Encoding train/val/test data into latent space...")
    Z_train = encoder.predict(X_train_scaled)
    Z_val = encoder.predict(X_val_scaled)
    Z_test = encoder.predict(X_test_scaled)

    print(f"Latent shapes – Train: {Z_train.shape}, Val: {Z_val.shape}, Test: {Z_test.shape}")

    # MLP classifier on latent features
    print("\nBuilding MLP classifier on latent features...")
    mlp = build_mlp_classifier(input_dim=code_dim, num_classes=NUM_PEOPLE)

    print("Training classifier...")
    clf_history = mlp.fit(
        Z_train, y_train,
        validation_data=(Z_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    # Plot classifier training curves
    plot_history(clf_history, title_prefix="MLP Classifier")

    # Evaluation on test set
    print("\nEvaluating on test set...")
    y_test_proba = mlp.predict(Z_test)
    y_test_pred = np.argmax(y_test_proba, axis=1)

    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred, digits=4))

    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix (Test):")
    print(cm)


    class_names = [f"Person {i}" for i in range(NUM_PEOPLE)]
    plot_confusion_matrix(cm, class_names=class_names, title="Test Confusion Matrix")

    # Show some sample test images with predictions
    print("\nShowing sample test predictions...")
    show_sample_predictions(X_test, y_test, y_test_pred, H, W, n_samples=8)


if __name__ == "__main__":
    main()
