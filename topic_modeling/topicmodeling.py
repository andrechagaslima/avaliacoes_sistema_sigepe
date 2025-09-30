from topicmodeling.BERTopic.BERTopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import TfidfVectorizer

def run_topic_modeling():
    save_data = 'data_topic_modeling'

    df = pd.read_csv('./data/dataFrame.csv')

    use_df = df[df['clean_text'].notna()]
    use_df = use_df.reset_index(drop=True)

    main_representation = KeyBERTInspired()

    # Additional ways of representing a topic
    aspect_model1 = PartOfSpeech("pt_core_news_sm")
    aspect_model2 = [KeyBERTInspired(top_n_words=10), MaximalMarginalRelevance(diversity=.3)]


    params = {
        'nr_topics': 10,
        'language': 'portuguese',
        'calculate_probabilities': True,
        'verbose': False,
        'top_n_words': 10,
        'representation_model' : {
    "Main": main_representation,
    "Aspect1":  aspect_model1,
    "Aspect2":  aspect_model2 
    },
        'hdbscan_model' : KMeans(n_clusters=10),
        #'ctfidf_model' : ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)
    }
    
    model = BERTopic(**params)

    model.fit(use_df['clean_text'])

    model.save(f"./topicmodeling/{save_data}/kmeans_1", serialization="pickle")

    model.dominant_topics(use_df['clean_text'], save_data, use_df['ID'].tolist())

    loaded_model = BERTopic.load(f"./topicmodeling/data_topic_modeling/kmeans_1")

    tfidf = TfidfVectorizer()
    tfidf.fit_transform(use_df['clean_text'].tolist())
    features = tfidf.get_feature_names_out()

    reverse_voc = {features[i]: i for i in range(len(features))}

    matrix = loaded_model.c_tf_idf_

    td = pd.read_csv('./topicmodeling/data_topic_modeling/Resumo_Topicos_Dominantes.csv')
    td = td.groupby('dominant_topic')
    td_dict = {topic: ids['papers'].tolist() for topic, ids in td}

    # para o topico 0
    infos_words_topics = {}
    for i in range(10):
        words_ids = matrix[i].nonzero()[1]
        words_scores = matrix[i].data
        dict_score = {column: value for column, value in zip(words_ids, words_scores)}
        infos_words_topics[i] = dict_score

    topicos = []
    docs = []
    docs_scores = []

    td = pd.read_csv('./topicmodeling/data_topic_modeling/Resumo_Topicos_Dominantes.csv')

    for i in range(len(use_df)):
        docs.append(use_df['ID'][i])
        dominant_topic = td['dominant_topic'][i]
        topicos.append(dominant_topic) # df de topicos dominantes
        tfidf = TfidfVectorizer()
        tfidf.fit_transform([use_df['clean_text'][i]])
        words = tfidf.get_feature_names_out()
        sum_score = 0
        for w in words:
            sum_score += infos_words_topics[dominant_topic][reverse_voc[w]]
        docs_scores.append(sum_score)

    data = {
        'dominant_topic': topicos,
        'document_id': docs,
        'document_score': docs_scores
    }
    df = pd.DataFrame(data)

    df.to_csv('./topicmodeling/data_topic_modeling/documents_scores.csv')

    save_data = 'data_num_topics'

    num_topics = 5

    params = {
        'nr_topics': num_topics,
        'language': 'portuguese',
        'calculate_probabilities': True,
        'verbose': False,
        'top_n_words': 10,
        'hdbscan_model' : KMeans(n_clusters=5)}

    model = BERTopic(**params)
    model.fit(use_df['clean_text'])

    model.save_txt(f'./topicmodeling/data_num_topics/{num_topics}/topics.txt')
    model.dominant_topics(use_df['clean_text'], f'./data_num_topics/{num_topics}/', use_df['ID'].tolist())
    model.save_json(f'./topicmodeling/data_num_topics/{num_topics}/topics_{num_topics}.json')

    df = pd.read_csv(f'./topicmodeling/data_num_topics/{num_topics}/Topicos_Dominantes.csv', sep='|')
    df['document_score'] = df[['Topico 0', 'Topico 1', 'Topico 2', 'Topico 3', 'Topico 4']].max(axis=1)
    result_df = df[['dominant_topic', 'id', 'document_score']]
    result_df.rename(columns={'id': 'document_id'}, inplace=True)
    result_df.to_csv(f'./topicmodeling/data_num_topics/{num_topics}/documents_scores.csv', index=False)

    num_topics = 10
    params = {
        'nr_topics': num_topics,
        'language': 'portuguese',
        'calculate_probabilities': True,
        'verbose': False,
        'top_n_words': 10,
        'hdbscan_model' : KMeans(n_clusters=num_topics)}
    model = BERTopic(**params)
    model.fit(use_df['clean_text'])

    model.save_txt(f'./topicmodeling/data_num_topics/{num_topics}/topics.txt')
    model.dominant_topics(use_df['clean_text'], f'./data_num_topics/{num_topics}/', use_df['ID'].tolist())
    model.save_json(f'./topicmodeling/data_num_topics/{num_topics}/topics_{num_topics}.json')

    df = pd.read_csv(f'./topicmodeling/data_num_topics/{num_topics}/Topicos_Dominantes.csv', sep='|')
    df['document_score'] = df[['Topico 0', 'Topico 1', 'Topico 2', 'Topico 3', 'Topico 4', 'Topico 5', 'Topico 6', 'Topico 7', 'Topico 8', 'Topico 9']].max(axis=1)
    result_df = df[['dominant_topic', 'id', 'document_score']]
    result_df.rename(columns={'id': 'document_id'}, inplace=True)
    result_df.to_csv(f'./topicmodeling/data_num_topics/{num_topics}/documents_scores.csv', index=False)

    num_topics = 15
    params = {
        'nr_topics': num_topics,
        'language': 'portuguese',
        'calculate_probabilities': True,
        'verbose': False,
        'top_n_words': 10,
        'hdbscan_model' : KMeans(n_clusters=num_topics)}
    model = BERTopic(**params)
    model.fit(use_df['clean_text'])

    model.save_txt(f'./topicmodeling/data_num_topics/{num_topics}/topics.txt')
    model.dominant_topics(use_df['clean_text'], f'./data_num_topics/{num_topics}/', use_df['ID'].tolist())
    model.save_json(f'./topicmodeling/data_num_topics/{num_topics}/topics_{num_topics}.json')

    df = pd.read_csv(f'./topicmodeling/data_num_topics/{num_topics}/Topicos_Dominantes.csv', sep='|')
    df['document_score'] = df[['Topico 0', 'Topico 1', 'Topico 2', 'Topico 3', 'Topico 4', 'Topico 5', 'Topico 6', 'Topico 7', 'Topico 8', 'Topico 9', 'Topico 10', 'Topico 11', 'Topico 12', 'Topico 13', 'Topico 14']].max(axis=1)
    result_df = df[['dominant_topic', 'id', 'document_score']]
    result_df.rename(columns={'id': 'document_id'}, inplace=True)
    result_df.to_csv(f'./topicmodeling/data_num_topics/{num_topics}/documents_scores.csv', index=False)

run_topic_modeling()