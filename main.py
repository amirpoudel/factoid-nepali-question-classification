import csv
import os
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configuration
INITIAL_LABELED = "Labeled.csv"
UNLABELED_POOL = "Unlabeled.csv"
ACTIVE_LABELS = "ActiveLearning_Labels.csv"
ACTIVE_UNLABELED = "Unlabeled_Active.csv"
CONFIDENCE_THRESHOLD = 0.8
MAX_ITERATIONS = 10
BATCH_SIZE = 5

def safe_file_copy(src, dst):
    if not os.path.exists(dst):
        with open(src, 'r', encoding='utf-8') as f1, \
             open(dst, 'w', encoding='utf-8') as f2:
            f2.write(f1.read())

def load_active_labels():
    if not os.path.exists(ACTIVE_LABELS):
        safe_file_copy(INITIAL_LABELED, ACTIVE_LABELS)
    
    with open(ACTIVE_LABELS, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load_active_unlabeled():
    if not os.path.exists(ACTIVE_UNLABELED):
        safe_file_copy(UNLABELED_POOL, ACTIVE_UNLABELED)
    
    with open(ACTIVE_UNLABELED, 'r', encoding='utf-8') as f:
        return [row['Questions'] for row in csv.DictReader(f) if 'Questions' in row]

def save_labels(data):
    with open(ACTIVE_LABELS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Questions", "CoarseType", "FineType", "WhWord"])
        writer.writeheader()
        writer.writerows(data)

def save_unlabeled(data):
    with open(ACTIVE_UNLABELED, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Questions"])
        for question in data:
            writer.writerow([question])

# Initialize Word2Vec model
word2vec_model = Word2Vec.load("./processed.word2vec")

def nepali_vectorizer(question, model):
    cleaned = question.replace('?', '').replace('"', '')
    tokens = cleaned.split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Prepare initial data
active_labeled = load_active_labels()
active_unlabeled = load_active_unlabeled()

# Initialize encoders for all labels
coarse_encoder = LabelEncoder()
fine_encoder = LabelEncoder()
wh_encoder = LabelEncoder()

# Fit encoders with initial data
coarse_types = [row["CoarseType"] for row in active_labeled]
fine_types = [row["FineType"] for row in active_labeled]
wh_words = [row["WhWord"] for row in active_labeled]

coarse_encoder.fit(coarse_types)
fine_encoder.fit(fine_types)
wh_encoder.fit(wh_words)


#  main active learning loop 
for iteration in range(MAX_ITERATIONS):
    print(f"\n{'='*40}")
    print(f"ITERATION {iteration+1}/{MAX_ITERATIONS}")
    print(f"{'='*40}\n")
    
    # Log training data stats
    print(f"[Training Data] Current size: {len(active_labeled)} samples")
    coarse_dist = pd.Series([row["CoarseType"] for row in active_labeled]).value_counts()
    print("Label Distribution:")
    print(coarse_dist.to_string())
    
    # Prepare training data
    X_train = np.array([nepali_vectorizer(row["Questions"], word2vec_model) for row in active_labeled])
    y_coarse = coarse_encoder.transform([row["CoarseType"] for row in active_labeled])
    y_fine = fine_encoder.transform([row["FineType"] for row in active_labeled])
    y_wh = wh_encoder.transform([row["WhWord"] for row in active_labeled])
    y_train = np.column_stack((y_coarse, y_fine, y_wh))

    # Train model
    classifier = MultiOutputClassifier(SVC(probability=True))
    classifier.fit(X_train, y_train)
    
    # Process unlabeled pool
    if not active_unlabeled:
        print("\n[Status] No more unlabeled samples remaining")
        break
        
    print(f"\n[Unlabeled Pool] Processing {len(active_unlabeled)} questions")
    
    # Vectorize questions
    X_unlabeled = np.array([nepali_vectorizer(q, word2vec_model) for q in active_unlabeled])
    
    # Get predictions and probabilities
    probabilities = [estimator.predict_proba(X_unlabeled) for estimator in classifier.estimators_]
    predictions = classifier.predict(X_unlabeled)
    
    # Calculate confidence scores
    confidences = np.min([np.max(proba, axis=1) for proba in probabilities], axis=0)
    
    # Decode predictions
    coarse_pred = coarse_encoder.inverse_transform(predictions[:, 0])
    fine_pred = fine_encoder.inverse_transform(predictions[:, 1])
    wh_pred = wh_encoder.inverse_transform(predictions[:, 2])
    
    # Detailed prediction logging
    print("\n[Sample Predictions]")
    for i, (q, c, f, w, conf) in enumerate(zip(
        active_unlabeled[:5],  # Show first 5 predictions
        coarse_pred[:5], 
        fine_pred[:5], 
        wh_pred[:5], 
        confidences[:5]
    )):
        print(f"\nQuestion {i+1}: {q}")
        print(f"Predicted: {c} > {f} | Wh: {w}")
        print(f"Confidence: {conf:.2f}")
        print(f"Probabilities:")
        for est_idx, (est_name, encoder) in enumerate(zip(
            ['Coarse', 'Fine', 'Wh'],
            [coarse_encoder, fine_encoder, wh_encoder]
        )):
            class_probs = probabilities[est_idx][i]
            top_idx = np.argmax(class_probs)
            print(f" - {est_name}: {encoder.classes_[top_idx]} ({class_probs[top_idx]:.2f})")

    # Confidence statistics
    print("\n[Confidence Analysis]")
    print(f"Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Minimum confidence: {np.min(confidences):.2f}")
    print(f"Maximum confidence: {np.max(confidences):.2f}")
    print(f"Mean confidence: {np.mean(confidences):.2f}")
    print(f"Questions above threshold: {np.sum(confidences >= CONFIDENCE_THRESHOLD)}")
    
    # Separate confident samples
    confident_mask = confidences >= CONFIDENCE_THRESHOLD
    new_entries = []
    
    # Log confident samples
    if np.any(confident_mask):
        print("\n[Confident Samples Added]")
        for q, c, f, w in enumerate(zip(
            np.array(active_unlabeled)[confident_mask],
            coarse_pred[confident_mask],
            fine_pred[confident_mask],
            wh_pred[confident_mask]
        )[:3]):  # Show first 3 added samples
            print(f"Added: '{q}'")
            print(f" - Coarse: {c}, Fine: {f}, Wh: {w}\n")
    else:
        print("\n[No Confident Samples Found] No questions met confidence threshold")

    # Update datasets
    new_entries = [{
        "Questions": q,
        "CoarseType": c,
        "FineType": f,
        "WhWord": w
    } for q, c, f, w in enumerate(zip(
        np.array(active_unlabeled)[confident_mask],
        coarse_pred[confident_mask],
        fine_pred[confident_mask],
        wh_pred[confident_mask]
    ))]
    
    active_labeled.extend(new_entries)
    active_unlabeled = list(np.array(active_unlabeled)[~confident_mask])
    
    # Save state
    save_labels(active_labeled)
    save_unlabeled(active_unlabeled)
    
    print(f"\n[Iteration Summary]")
    print(f"Added: {len(new_entries)} new samples")
    print(f"Remaining unlabeled: {len(active_unlabeled)}")
    print(f"New training set size: {len(active_labeled)}")
    print(f"{'='*40}\n")

print("\n[Final Summary]")
print(f"Total iterations completed: {MAX_ITERATIONS}")
print(f"Final training set size: {len(active_labeled)}")
print(f"Final unlabeled pool size: {len(active_unlabeled)}")

# # Main active learning loop
# for iteration in range(MAX_ITERATIONS):
#     print(f"\n=== Iteration {iteration+1}/{MAX_ITERATIONS} ===")
    
#     # Prepare multi-label training data
#     X_train = np.array([nepali_vectorizer(row["Questions"], word2vec_model) for row in active_labeled])
#     y_coarse = coarse_encoder.transform([row["CoarseType"] for row in active_labeled])
#     y_fine = fine_encoder.transform([row["FineType"] for row in active_labeled])
#     y_wh = wh_encoder.transform([row["WhWord"] for row in active_labeled])
#     y_train = np.column_stack((y_coarse, y_fine, y_wh))

#     # Train multi-output classifier
#     classifier = MultiOutputClassifier(SVC(probability=True))
#     classifier.fit(X_train, y_train)
#     print(f"Model trained with {len(active_labeled)} samples")

#     # Process unlabeled pool
#     if not active_unlabeled:
#         print("No more unlabeled samples")
#         break

#     # Vectorize unlabeled questions
#     X_unlabeled = np.array([nepali_vectorizer(q, word2vec_model) for q in active_unlabeled])
    
#     # Get predictions and probabilities
#     probabilities = [estimator.predict_proba(X_unlabeled) for estimator in classifier.estimators_]
    
#     # Calculate confidence scores (minimum confidence across all three labels)
#     confidences = np.min([
#         np.max(proba, axis=1) 
#         for proba in probabilities
#     ], axis=0)
    
#     # Get predicted labels
#     predictions = classifier.predict(X_unlabeled)
#     coarse_pred = coarse_encoder.inverse_transform(predictions[:, 0])
#     fine_pred = fine_encoder.inverse_transform(predictions[:, 1])
#     wh_pred = wh_encoder.inverse_transform(predictions[:, 2])

#     # Separate confident and unconfident samples
#     confident_mask = confidences >= CONFIDENCE_THRESHOLD
#     confident_questions = np.array(active_unlabeled)[confident_mask]
#     confident_coarse = coarse_pred[confident_mask]
#     confident_fine = fine_pred[confident_mask]
#     confident_wh = wh_pred[confident_mask]

#     # Update labeled data
#     new_entries = [
#         {
#             "Questions": q,
#             "CoarseType": c,
#             "FineType": f,
#             "WhWord": w
#         } for q, c, f, w in zip(confident_questions, confident_coarse, confident_fine, confident_wh)
#     ]
#     active_labeled.extend(new_entries)
    
#     # Update unlabeled pool
#     active_unlabeled = list(np.array(active_unlabeled)[~confident_mask])
    
#     # Save progress
#     save_labels(active_labeled)
#     save_unlabeled(active_unlabeled)
    
#     print(f"Added {len(new_entries)} confident samples")
#     print(f"Remaining unlabeled: {len(active_unlabeled)}")

# print("\nActive learning completed!")
# print(f"Final training set: {len(active_labeled)} samples")
# print(f"Final unlabeled pool: {len(active_unlabeled)} samples")