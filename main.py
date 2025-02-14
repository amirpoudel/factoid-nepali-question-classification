from gensim.models import Word2Vec # type: ignore
import json
import numpy as np # type: ignore

# Load labeled data
with open('./dataset/sample_label_data.json', 'r', encoding='utf-8') as f:
    labeled_data = json.load(f)

# Load unlabeled data
with open('./dataset/sample_unlable_data.json', 'r', encoding='utf-8') as f:
    unlabeled_data = json.load(f)

# need to work on this later
# def preprocess_question(question):
#     # Normalize (lowercase, remove non-alphabetic characters)
#     question = re.sub(r'[^a-zA-Z0-9\s]+', '', question.lower())
    
#     # Tokenize
#     tokens = word_tokenize(question)
    
#     # Apply stemming
#     tokens_stemmed = [stemmer.stem(word) for word in tokens]
    
#     return tokens_stemmed

# # Apply preprocessing to your labeled dataset
# labeled_data_processed = []
# for entry in labeled_data:
#     processed_question = preprocess_question(entry["question"])
#     labeled_data_processed.append(processed_question)

# print(labeled_data_processed)


model = "./processed.word2vec"
word2vec_model = Word2Vec.load(model)



# Function to get sentence vector (by averaging word vectors)
def get_question_vector(question, word2vec_model):
    # Preprocess the question (tokenize and stem)
    processed_question = question
    
    # Get word vectors for each token in the question
    word_vectors = []
    for word in processed_question:
        if word in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[word])
    
    # If there are no word vectors (no words in Word2Vec model), return zero vector
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

# Example usage
question = "नेपालको पहिलो राष्ट्रपति को हुन्?"
question_vector = get_question_vector(question, word2vec_model)
print(question_vector)







