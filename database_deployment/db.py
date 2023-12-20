import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import csv


chroma_client = chromadb.Client()

collection_cS = chroma_client.create_collection(name='colletion_cS')
collection_lc = chroma_client.create_collection(name='colletion_lc')

collection_cS.add(
    documents = ['Boilerplate document', 'Boilerplate document 2'],
    metadatas = [{'source': 'my_source'}, {'source_2','my_source_2'}],
    ids = ['id1', 'id2']
)

collection_lc.add(
    documents = ['Boilerplate document', 'Boilerplate document 2'],
    metadatas = [{'source': 'my_source'}, {'source_2','my_source_2'}],
    ids = ['id1', 'id2']
)

results_cS = collection_cS.query(
    query_texts = ['This is a query document'],
    n_results = 10,
    include = ['distances', 'metadatas', 'embeddings', 'documents']
)

print(results_cS)

results_lc = collection_lc.query(
    query_texts= ['This is a lc document'],
    n_results= 10,
    include = []
)

print(results_lc)