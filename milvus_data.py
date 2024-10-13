from pymilvus import connections
from pymilvus import utility
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
import pandas as pd
import ollama


MILVUS_HOST = "localhost"
MILVUS_PORT = 19530


def find_most_similar_sentence(symptoms_input, top_k=3):
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection_name = "disease_diagnosis"
    collection = Collection(name=collection_name)
    collection.load()
    response = ollama.embeddings(model="mxbai-embed-large", prompt=symptoms_input)
    symtoms_embedding = response["embedding"]
    symtoms_embedding = [symtoms_embedding]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    output_fields = [
        "disease_name",
        "symptoms",
        "cure",
        "causes",
        "precautions",
        "medicines",
    ]
    result = collection.search(
        data=symtoms_embedding,
        anns_field="symptoms_embedding",
        param=search_params,
        limit=top_k,
        output_fields=output_fields,
    )
    top_n_results = []
    for hits in result:
        for hit in hits:
            top_n_results.append({
                'Disease name': hit.entity.disease_name,
                'Symptoms': hit.entity.symptoms,
                'Cure': hit.entity.cure,
                'Causes': hit.entity.causes,
                'Precautions': hit.entity.precautions,
                'Medicines': hit.entity.medicines,
            })
    return top_n_results
