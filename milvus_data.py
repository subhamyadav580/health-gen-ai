from pymilvus import connections
from pymilvus import utility
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
import pandas as pd
import ollama
import json
from neo4j import GraphDatabase
import uuid



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


def make_patient_specific_nodes_relations(input_patient_history):
    pass

    # model = OllamaLLM(model="llama3.1")

def parse_medical_history(medical_history_text):
    
    prompt = """
    Extract structured information from the following {Medical History} realated to {patient_id} text in the format given below :

    Structure the information extracted above as JSON with the following format:
    {{
        "Patient": {{
            "id": "patient_id",
            "age": Age,
            "gender": "Gender",
            "place_of_birth":"Place Of Birth"
        }},
        "Diseases": ["Disease1", "Disease2", ...],
        "Symptoms": ["Symptom1", "Symptom2", ...],
        "Medications": ["Medication1", "Medication2", ...],
        "Doctors": ["Doctor1", "Doctor2", ...],
        "Procedures": ["Procedure1", "Procedure2", ...],
        "Genetic Disorders":["Disorder1","Disorder2" ,....],
        "Allergies Information":["Allergy1", "Allergy2" ,.....]
    }}
    """


    
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | model
    patient_id = uuid
    res = chain.invoke({"Medical History" : medical_history_text.get("medical_history"), "patient_id":medical_history_text.get("patient_id")})


    structured_data = res
    return structured_data




def store_medical_data_in_graph_db(structured_data_json):

    try:
        driver = GraphDatabase.driver("bolt://172.234.22.188:7687", auth=("neo4j", "qwerty1234"))
        data = structured_data_json
        patient = data['Patient']
        diseases = data.get('Diseases', [])
        symptoms = data.get('Symptoms', [])
        medications = data.get('Medications', [])
        doctors = data.get('Doctors', [])
        procedures = data.get('Procedures', [])
        Allergies = data.get('Allergies Information',[])
        # Generate a random UUID
        patient_id = str(uuid.uuid4())
        
        with driver.session() as session:
            # Create Patient Node
            session.run("""
                MERGE (p:Patient {id: $id})
                SET p.age = $age,
                    p.gender = $gender
            """, id=patient_id,  age=patient['age'], gender=patient['gender'])
            
            # Create Disease Nodes and Relationships
            for disease in diseases:
                session.run("""
                    MERGE (d:Disease {name: $disease})
                    MERGE (p:Patient {id: $id})
                    MERGE (p)-[:HAS_DISEASE]->(d)
                """, disease=disease, id=patient_id)
            
            # Create Symptom Nodes and Relationships
            for symptom in symptoms:
                session.run("""
                    MERGE (s:Symptom {name: $symptom})
                    MERGE (p:Patient {id: $id})
                    MERGE (p)-[:EXHIBITS]->(s)
                """, symptom=symptom, id=patient_id)
            
            # Create Medication Nodes and Relationships
            for medication in medications:
                session.run("""
                    MERGE (m:Medication {name: $medication})
                    MERGE (p:Patient {id: $id})
                    MERGE (p)-[:TAKES]->(m)
                """, medication=medication, id=patient_id)
            
            # Create Doctor Nodes and Relationships
            for doctor in doctors:
                session.run("""
                    MERGE (doc:Doctor {name: $doctor})
                    MERGE (p:Patient {id: $id})
                    MERGE (p)-[:CONSULTED_BY]->(doc)
                """, doctor=doctor, id=patient_id)
            
            # Create Procedure Nodes and Relationships
            for procedure in procedures:
                session.run("""
                    MERGE (proc:Procedure {name: $procedure})
                    MERGE (p:Patient {id: $id})
                    MERGE (p)-[:UNDERGOES]->(proc)
                """, procedure=procedure, id=patient_id)
    except:
        pass
