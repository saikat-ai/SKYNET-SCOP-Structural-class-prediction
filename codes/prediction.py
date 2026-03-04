import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load saved objects (assuming pre-trained models, tokenizer, and scaler)
rf = pickle.load(open("rf_model.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

MAX_LENGTH = 500  # Must match training setup
K = 6             # 6-mer


# Function to convert a sequence to feature vector (extract 6-mers)
def get_kmers(seq, k=6):
    seq = str(seq).lower()
    if len(seq) < k:
        return [seq]  # fallback: use full seq as single token
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def sequence_to_features(sequence, tokenizer, scaler, max_length=500, k=6):
    text = " ".join(get_kmers(sequence, k=k))
    seqs = tokenizer.texts_to_sequences([text])
    X = pad_sequences(seqs, maxlen=max_length, padding="post", truncating="post")
    X_scaled = scaler.transform(X)
    return X_scaled


# ===== Compute prediction using Random Forest (RF) =====
def predict_class(sequence):
    # Extract features using pre-trained tokenizer and scaler
    X_scaled = sequence_to_features(sequence, tokenizer, scaler, max_length=MAX_LENGTH, k=K)

    # Get probabilities from Random Forest model
    proba_rf = rf.predict_proba(X_scaled)
    
    # Get the predicted class (index of highest probability)
    pred_idx = np.argmax(proba_rf, axis=1)[0]

    # Decode the predicted class index to the original label (e.g., 1, 2, 3, 4)
    class_label = label_encoder.inverse_transform([pred_idx])[0]
    
    return class_label, proba_rf


# ===== Extract top important 6-mers based on Random Forest feature importance =====
def extract_top_important_kmers(sequence, tokenizer, rf, scaler, top_n=15):
    # Extract 6-mers from the sequence
    sequence_kmers = get_kmers(sequence, k=K)

    # Get Random Forest feature importances
    rf_importances = rf.feature_importances_

    # Sort the features based on Random Forest feature importance
    sorted_kmers_rf = sorted(zip(sequence_kmers, rf_importances), key=lambda x: x[1], reverse=True)[:top_n]

    return sorted_kmers_rf


# Interactive session for prediction and feature importance using Random Forest (RF)
if __name__ == "__main__":
    print("=== Sequence Classifier (Random Forest + Feature Importance, 6-mers) ===")
    print("Type a sequence and press Enter to get class (1/2/3/4).")
    print("Type 'q' or just press Enter to quit.\n")

    while True:
        seq = input("Enter sequence: ").strip()
        if seq.lower() == "q" or seq == "":
            print("Exiting.")
            break

        try:
            # Predict class using Random Forest (RF)
            pred_class, proba_rf = predict_class(seq)
            print(f"Predicted Class: {pred_class}\n")

            # Extract top important 6-mers using Random Forest's feature importance
            sorted_relevant_features_rf = extract_top_important_kmers(seq, tokenizer, rf, scaler, top_n=15)

            # Print the top features based on Random Forest's feature importance
            print("\nTop 5 important 6-mers based on Random Forest's feature importances:")
            for km, importance in sorted_relevant_features_rf:
                print(f"6-mer: {km} with RF Importance: {importance:.6f}")

        except Exception as e:
            print(f"Error during prediction: {e}\n")
