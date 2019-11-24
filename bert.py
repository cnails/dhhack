import h5py
import keras
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from keras_bert.layers import MaskedGlobalMaxPool1D
from sklearn.metrics.pairwise import cosine_similarity

with open('all_sents.txt', 'r', encoding='utf-8') as f:
    sents = f.read().split('\n')

model_path = 'bert'
paths = get_checkpoint_paths(model_path)
seq_len = 50
bert_model = load_trained_model_from_checkpoint(config_file=paths.config, checkpoint_file=paths.checkpoint, seq_len=seq_len)
pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(bert_model.output)
bert_model = keras.models.Model(inputs=bert_model.inputs, outputs=pool_layer)
token_dict = load_vocabulary(paths.vocab)
tokenizer = Tokenizer(token_dict)


def get_bert_vector(doc, bert_model, token_dict, seq_len):
    tokenized_doc = tokenizer.tokenize(doc)[:seq_len]
    segments = [0] * seq_len
    indices = [token_dict[elem] for elem in tokenized_doc]
    indices += [0] * (seq_len - len(indices))
    return bert_model.predict([np.array([indices]), np.array([segments])])[0]


def get_most_probable_docs(doc_vector, docs_matrix):
    cosine_values = cosine_similarity(docs_matrix, doc_vector.reshape(1, -1)).reshape(docs_matrix.shape[0])
    return np.random.choice([sents[sent_id] for sent_id, _ in sorted(list(
        enumerate(cosine_values)), key=lambda elem: elem[1], reverse=True)][:10])


def search_bert(query, bert_vectors, bert_model, token_dict, seq_len):
    bert_vector = get_bert_vector(query, bert_model, token_dict, seq_len)
    return get_most_probable_docs(bert_vector, bert_vectors)


h5f = h5py.File('bert.h5', 'r')
bert_vectors = h5f['dataset_1'][:]
h5f.close()

print(search_bert('Мне страшно жить здесь', bert_vectors, bert_model, token_dict, seq_len))
