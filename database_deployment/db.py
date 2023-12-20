import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import csv


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')


with open('item.csv') as file:
    lines = csv.reader(file)

    documents = []
    metadatas = []
    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i == 0:
            continue
        
        documents.append(line[1])
        metadatas.append({'item_id': line[0]})
        ids.append(str(id))
        id += 1

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name='sample_1', embedding_function=sentence_transformer_ef)

collection.add(
    documents = ['Boilerplate document', 'Boilerplate document 2'],
    metadatas = [{'source': 'my_source'}, {'source_2','my_source_2'}],
    ids = ['id1', 'id2']
)

results = collection.query(
    query_texts = ['This is a query document'],
    n_results = 1,
    include = ['distances', 'metadatas', 'embeddings', 'documents']
)

results