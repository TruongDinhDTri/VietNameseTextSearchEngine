import os
import numpy as np
import nltk
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from underthesea import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('punkt')

with open('vietnamese-stopwords-dash.txt', mode='r', encoding='utf-8') as f:
    stop_words = set(f.read().split())


def get_tokenized_list_vi(doc_text):
    tokens = word_tokenize(doc_text, format="text")
    tokens = nltk.word_tokenize(tokens)
    return tokens


def remove_stopwords_vi(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words)
    return cleaned_text


def Clean_Data(input_text):
    input_text = get_tokenized_list_vi(input_text)
    input_text = remove_stopwords_vi(input_text)
    q = ' '.join(input_text)
    return q


def Run_Model(input_text):
    cleaned_input = Clean_Data(input_text)
    model_path = os.getcwd() + '/tf_idf_vietnamese_full.sav'
    tf_idf = pickle.load(open(model_path, 'rb'))
    query_vector = tf_idf.transform([cleaned_input])
    vector_path = os.getcwd() + '/doc_vector_vietnamese_full.npz'
    doc_vectors = sparse.load_npz(vector_path)
    cosineSimilarities = cosine_similarity(doc_vectors, query_vector).flatten()
    related_docs_indices = cosineSimilarities.argsort()[:-21:-1]
    return related_docs_indices


def Read_Content(the_input):
    all_content = {}
    indices = Run_Model(the_input)
    for each_index in indices:
        path = os.getcwd()
        path = path + '/news_dataset/' + file_names[each_index]
        file = open(path, "rb")
        content_in_file = file.read()
        all_content[file_names[each_index]] = content_in_file
    return all_content


def Query_Expansion(token, old_query, transformer, matrix_normalized, range_qe=1):
    token = token.lower()
    if np.where(transformer == token)[0].size == 0:
        return []
    idx = np.where(transformer == token)[0][0]
    list_expansion = []
    for pos in matrix_normalized[idx].argsort()[:-21:-1]:
        if transformer[pos] not in old_query:
            list_expansion.append(transformer[pos])
    return list_expansion[0:range_qe]


def Update_Query(user_input, retrieved_content):
    user_input = get_tokenized_list_vi(user_input)
    user_input = remove_stopwords_vi(user_input)
    vectorizer_qe = CountVectorizer()
    X_qe = vectorizer_qe.fit_transform(retrieved_content)
    m = X_qe.toarray().T
    s = m @ m.T
    s_nor = np.zeros(s.shape)
    # unnomalize
    # s_nor = s.copy()
    row, col = s.shape
    for u in range(row):
        for v in range(col):
            s_nor[u][v] = s[u][v] / (s[u][u] + s[v][v] - s[u][v])
    final_query = []
    get_feature_names_out = vectorizer_qe.get_feature_names()
    get_feature_names_out = np.array(get_feature_names_out)
    for qu in user_input:
        final_query.append(qu)
        final_query = final_query + Query_Expansion(qu, user_input + final_query, get_feature_names_out, s_nor, range_qe=1)
    updated = ' '.join(final_query)
    return updated


def Evaluate_System(queries):
    count = 1
    precisions_list = []
    for each_query in queries:
        # Get the predicted result
        relevant_indices = Run_Model(each_query)

        # Get the ground truth (really is relevant)
        path_of_ground_truth = os.getcwd() + '/ground_truth/ground_truth/' + 'gt_' + str(count) + '.txt'
        ground_truth_files = open(path_of_ground_truth, "r")
        ground_truth_content = ground_truth_files.read()
        ground_truth_indices = ground_truth_content.split('\n')
        true_relevant = 0

        # Fix some bug in creating ground truth file
        ground_truth_indices = [i for i in ground_truth_indices if i != '']
        ground_truth_indices = [int(i) for i in ground_truth_indices]
        for each_predicted in relevant_indices:
            if each_predicted in ground_truth_indices:
                true_relevant += 1
        precisions_list.append(true_relevant / 20)  # just consider top 20 results
        count += 1
    return precisions_list


file_names = np.load("file_names_vietnamese_full.npy")

# Prepare for evaluating model
query = open('ground_truth/list_query.txt', "r", encoding='utf-8')
content = query.read()
arr_queries = content.split('\n')
arr_queries = [text for text in arr_queries if text != '']

# Get the result and display it
precisions = Evaluate_System(arr_queries)
table = {'Từ khoá': arr_queries,
         'Precision': precisions
         }
table = pd.DataFrame(data=table)
table = table.set_index(pd.Index([i for i in range(1, 11)]))


