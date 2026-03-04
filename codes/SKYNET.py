import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, precision_score,recall_score, f1_score, confusion_matrix)
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D,Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)
num_classes = len(np.unique(y_encoded))
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,test_size=0.2, stratify=y_encoded, random_state=42)
# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== Prepare CNN Input ==========
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0],
X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0],
X_test_scaled.shape[1], 1)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
# ========== Define Hybrid Loss ==========
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return loss

def hybrid_loss(alpha=0.7, beta=0.3, gamma=2.0):
    ce = CategoricalCrossentropy()
    fl = focal_loss(gamma=gamma)
    def loss_fn(y_true, y_pred):
        return alpha * ce(y_true, y_pred) + beta * K.mean(fl(y_true, y_pred))
    return loss_fn
# ========== CNN Model ==========
cnn_model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss=hybrid_loss(alpha=0.6,
beta=0.3), metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=16, verbose=0)

# ========== Random Forest ==========
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ========== LightGBM ==========
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train_scaled, y_train)

# ========== Predict Probabilities ==========
cnn_probs = cnn_model.predict(X_test_cnn)
rf_probs = rf_model.predict_proba(X_test_scaled)
lgb_probs = lgb_model.predict_proba(X_test_scaled)

# ========== Weighted Ensemble ==========
ensemble_probs = (
    0.85 * cnn_probs +
    1.00 * rf_probs +
    0.90 * lgb_probs
) / (1.00 + 0.90 + 0.75)

y_pred = np.argmax(ensemble_probs, axis=1)
# ========== Evaluation ==========
target_names = [str(cls) for cls in le.classes_]
print("Classification Report:\n", classification_report(y_test,y_pred, target_names=target_names))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
