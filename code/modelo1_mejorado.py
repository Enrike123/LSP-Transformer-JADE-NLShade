# -*- coding: utf-8 -*-
import os
import time
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (LayerNormalization, Dense, Dropout,
                                     MultiHeadAttention, Add,
                                     GlobalAveragePooling1D, Input)
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
plt.switch_backend("agg")  # evita bloquear en servidores/CLI

# =========================
# Utilidades y configuración
# =========================

def set_seeds(seed: int = 42):
    import random
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def load_data(file_path):
    print(f"Cargando dataset desde: {file_path}")
    dataset = np.load(file_path, allow_pickle=True).item()
    data = dataset["data"]
    labels = dataset["labels"]
    print(f"Forma X: {data.shape}  | y: {labels.shape}")
    return data, labels

def split_data(data, labels, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"Train: {X_train.shape}, {y_train.shape} | Val: {X_val.shape}, {y_val.shape}")
    return X_train, X_val, y_train, y_val

# =================
# Positional encoding
# =================

def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]       # (L, 1)
    dimensions = np.arange(d_model)[np.newaxis, :]      # (1, D)
    angles = positions / np.power(10000, (2 * (dimensions // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles, dtype=tf.float32)            # (L, D)

# ====================
# Bloque Transformer
# ====================

def transformer_encoder(inputs, nhead, nhid, dropout_rate):
    """
    inputs: (B, L, D)
    nhead: número de cabezas
    nhid: tamaño del FFN
    """
    d_model = int(inputs.shape[-1])
    key_dim = max(1, d_model // int(nhead))  # <- ajuste clave para MHA
    attn = MultiHeadAttention(num_heads=nhead, key_dim=key_dim)(inputs, inputs)
    attn = Dropout(dropout_rate)(attn)
    attn = Add()([inputs, attn])
    attn = LayerNormalization()(attn)

    ffn = Dense(nhid, activation='relu')(attn)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out = Add()([attn, ffn])
    return LayerNormalization()(out)

def build_transformer_model(input_shape, num_classes, nlayers, nhead, nhid, dropout_rate):
    """
    input_shape: (L, D)
    """
    inputs = Input(shape=input_shape)
    L, D = input_shape

    # Positional encoding dinámico
    pe = positional_encoding(L, D)           # (L, D)
    pe = tf.expand_dims(pe, axis=0)          # (1, L, D)
    x = Add()([inputs, pe])

    for _ in range(nlayers):
        x = transformer_encoder(x, nhead, nhid, dropout_rate)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# ==========================
# Scheduler y entrenamiento
# ==========================

def lr_schedule(epoch, lr):
    return lr * 0.5 if epoch > 20 else lr

def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    # por defecto
    return tf.keras.optimizers.Adam(learning_rate=lr)

def train_transformer_model(
    X_train, y_train, X_val, y_val, num_classes,
    nlayers=6, nhead=8, nhid=1024, dropout_rate=0.3,
    lr=1e-4, batch_size=32, epochs=12, patience=4,
    optimizer_name="adam", input_shape=(60, 96),
    verbose=0
):
    """
    Entrenamiento 'corto' (warm-up) para evaluación rápida.
    Devuelve: (f1_macro_val, model, history, elapsed_seconds)
    """
    t0 = time.time()
    model = build_transformer_model(input_shape, num_classes, nlayers, nhead, nhid, dropout_rate)
    opt = make_optimizer(optimizer_name, lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    lrs = LearningRateScheduler(lr_schedule)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[es, lrs],
        verbose=verbose
    )

    # Macro-F1 de validación
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    f1 = f1_score(y_val, y_pred, average='macro')

    elapsed = time.time() - t0
    return f1, model, history, elapsed

def train_full_and_save(
    X_train, y_train, X_val, y_val, num_classes, hp: dict,
    input_shape=(60, 96), out_prefix="transformer_model_sequence"
):
    """
    Re-entrena extendido con los mejores hiperparámetros y guarda artefactos.
    """
    f1, model, history, _ = train_transformer_model(
        X_train, y_train, X_val, y_val, num_classes,
        nlayers=hp['nlayers'], nhead=hp['nhead'], nhid=hp['nhid'],
        dropout_rate=hp['dropout'], lr=hp['lr'], batch_size=hp['batch_size'],
        epochs=hp.get('epochs_full', 30), patience=hp.get('patience_full', 6),
        optimizer_name=hp.get('opt_name', 'adam'),
        input_shape=input_shape, verbose=1
    )

    # Curvas de aprendizaje
    plot_learning_curves(history, fname=f"{out_prefix}_learning_curve.png")

    # Guardar modelo y HP
    model_path = f"{out_prefix}_optimized.h5"
    model.save(model_path)
    with open(f"{out_prefix}_best_hp.json", "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2)
    print(f"Modelo guardado en: {model_path}")
    return f1, model, history

# =======================
# Curvas y evaluación
# =======================

def plot_learning_curves(history, fname="learning_curve_transformer.png"):
    epochs = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", marker="s")
    plt.title("Transformer Learning Curves")
    plt.ylabel("Loss"); plt.xlabel("Epochs")
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, label="Train Acc", marker="o")
    plt.plot(epochs, val_acc, label="Val Acc", marker="s")
    plt.ylabel("Accuracy"); plt.xlabel("Epochs")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    print(f"Curvas de aprendizaje guardadas en: {fname}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

# ==========
# Ejemplo CLI
# ==========

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="gesture_dataset12clases_5aum.npy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_file = args.data if os.path.isabs(args.data) else os.path.join(base_path, args.data)

    data, labels = load_data(dataset_file)
    X_train, X_val, y_train, y_val = split_data(data, labels, test_size=0.2, random_state=args.seed)

    num_classes = len(np.unique(labels))

    # Entrenamiento rápido (warm-up) con valores por defecto
    f1, model, history, elapsed = train_transformer_model(
        X_train, y_train, X_val, y_val, num_classes,
        nlayers=6, nhead=8, nhid=1024, dropout_rate=0.3,
        lr=1e-4, batch_size=32, epochs=12, patience=4,
        optimizer_name="adam", input_shape=(X_train.shape[1], X_train.shape[2]),
        verbose=1
    )
    print(f"[DEMO] F1 macro validación: {f1:.4f}  | tiempo: {elapsed:.1f}s")
    evaluate_model(model, X_val, y_val)
