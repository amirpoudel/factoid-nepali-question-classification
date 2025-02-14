from gensim.models import Word2Vec # type: ignore
import json

# Load labeled data
with open('./dataset/sample_label_data.json', 'r', encoding='utf-8') as f:
    labeled_data = json.load(f)

# Load unlabeled data
with open('./dataset/sample_unlable_data.json', 'r', encoding='utf-8') as f:
    unlabeled_data = json.load(f)

# Print data to check
# print(labeled_data)
# print(unlabeled_data)


model = "./processed.word2vec"
word2vec_model = Word2Vec.load(model)

def question_to_vector(word):
    return word2vec_model.wv[word]

print(question_to_vector("नेपाल"))





