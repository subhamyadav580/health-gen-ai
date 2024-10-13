from fastapi import FastAPI, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from milvus_data import find_most_similar_sentence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from langchain_openai import ChatOpenAI
import shutil

from image_model import CNNModel
import torch
from PIL import Image
import torchvision.transforms as transforms


# os.environ["OPENAI_API_KEY"] = ("*****")

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "uploaded_images/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists


model = OllamaLLM(model="llama3.1")
# model = ChatOpenAI(model="gpt-3.5-turbo")


image_model = CNNModel()
image_model.load_state_dict(
    torch.load(
        "/Users/shubhamyadav/Documents/Personal/chest_xray_cnn_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)  # Load the model state
image_model.eval()


# Diagnosis template for processing
DIAGNOSIS_TEMPLATE = """
You are an AI model that aids in diagnosing medical conditions using text data. 
Your role is to analyze the provided report data, compare it with relevant information from the knowledge base, 
and deliver a clear and comprehensive explanation of the patient's condition.

Consider the following:
1. Medical reports may contain complex terminology; simplify your explanations when necessary to enhance understanding.
2. You will receive multiple possible conditions based on the report. Compare these options and select the most appropriate diagnosis.
3. If details in the report are unclear, use relevant medical knowledge and the retrieved information to provide an informed diagnosis.
4. Check the accuracy of the retrieved data against the user input. If the information seems incorrect or irrelevant, clarify this in your response.

User Input: {report_data}

Relevant Medical Documents (Top 3):
{retrieved_data}

### Processing Logic:
1. **Greeting Response**: 
   - If the User Input is a greeting (e.g., "Hi", "Hello", "Hey"), respond warmly: 
     "Hello! How can I assist you today? If you have any medical questions or need help with a patient report, feel free to share!"

2. **Non-Medical Inquiry Response**: 
   - If the User Input is not related to medical conditions or patient reports, respond with:
     "Thank you for your message! However, I can only assist with medical-related inquiries. Please provide relevant medical details or questions for further assistance."

3. **Medical Inquiry Processing**: 
   - If the User Input relates to a medical condition or patient report, proceed with the analysis as follows:
   - **Diagnosis**: Provide the most accurate diagnosis based on your analysis of the user input and the retrieved documents.
   - **Key Insights**: Relevant findings or conditions inferred from the medical data.
   - **Potential Symptoms**: Likely symptoms related to the condition, explicitly mentioned in the report or inferred from the retrieved data.
   - **Recommended Next Steps**: Suggestions for further tests, treatments, or follow-up actions based on the diagnosis.
   - **Relevant Medications**: If applicable, list general medication examples related to the diagnosis. If unsure, refrain from providing medications.
"""

IMAGE_DIAGNOSIS_TEMPLATE = """
You are an AI model that aids in diagnosing medical conditions based on image classification and relevant medical documents. 
Your role is to analyze the provided image classification label, compare it with relevant information from the knowledge base, 
and deliver a clear and comprehensive explanation of the patient's condition.

Consider the following:
1. The input consists of an image classification label (e.g., a chest x-ray result) and optional patient history or user input. 
2. Use the classified label as the primary basis for diagnosis, supplemented by relevant medical knowledge.
3. If user input is unclear or inconsistent with the label, provide an informed diagnosis based on the label and top documents.
4. Deliver a simple and comprehensive explanation of the condition, considering the patient's details when provided.

User Input (Optional): {user_input}

### Image Classification:
The image provided has been classified with the following label: **{disease_label}**.

### Relevant Medical Documents (Top 3):
{retrieved_data}

### Processing Logic:
1. **Greeting Response**:
   - If the User Input is a greeting (e.g., "Hi", "Hello", "Hey"), respond warmly: 
     "Hello! How can I assist you today? If you have any medical questions or need help interpreting an image report, feel free to share!"

2. **Non-Medical Inquiry Response**:
   - If the User Input is not related to medical conditions or image reports, respond with:
     "Thank you for your message! However, I can only assist with medical-related inquiries. Please provide relevant medical details or questions for further assistance."

3. **Medical Inquiry Processing**:
   - If the User Input relates to the provided image or patient condition, proceed with the analysis:
   - **Diagnosis**: Provide the most accurate diagnosis based on the classified label and the retrieved medical documents.
   - **Key Insights**: Highlight any relevant information or conditions inferred from the image label and top documents.
   - **Potential Symptoms**: List likely symptoms related to the classified condition, inferred from the label or medical documents.
   - **Recommended Next Steps**: Suggest additional diagnostic tests, treatments, or follow-up actions based on the classified condition.
   - **Relevant Medications**: If applicable, list common medications associated with the condition. If unsure, refrain from providing medications.

### Example Response:
- **Diagnosis**: Based on the chest x-ray image, the most likely condition is pneumonia, as classified by the model. The top medical documents support this diagnosis, citing similar symptoms and image findings. However, the possibility of bronchitis or asthma should be kept in mind if other symptoms are present.
- **Key Insights**: The image shows lung opacities consistent with bacterial pneumonia, typically caused by an infection. Top documents reinforce this conclusion, referencing similar imaging patterns.
- **Potential Symptoms**: Common symptoms of pneumonia include cough, fever, and difficulty breathing. The patient may experience chest pain or fatigue as well.
- **Recommended Next Steps**: A follow-up chest X-ray and blood culture would help confirm the diagnosis. Sputum analysis may also be helpful in identifying the bacterial cause.
- **Relevant Medications**: Antibiotics such as amoxicillin or a macrolide like azithromycin are often prescribed for bacterial pneumonia, though a medical consultation is required to confirm the treatment.
"""


SYMPTOMS_TEMPLATE = """
You are an AI assistant that provides medical information. 
Your role is to generate a comprehensive description of the symptoms for a given medical condition based on the disease name.

**Disease Name**: {disease_name}

### Task:
1. Using your medical knowledge, provide a detailed paragraph describing the typical symptoms associated with the disease {disease_name}.
2. Be clear and concise in your explanation. If relevant, include information on early symptoms, progression, and any severe or warning signs that may require immediate medical attention.
3. Avoid using overly complex medical terms without explaining them. Make the description accessible to a general audience.

### Example Output:
Pneumonia typically presents with symptoms such as a persistent cough, often producing mucus or phlegm, fever, chills, and difficulty breathing. In severe cases, chest pain may be felt during breathing or coughing. Fatigue, sweating, and a rapid heartbeat can also accompany pneumonia. Some people, especially older adults or those with weakened immune systems, may experience confusion or changes in mental awareness. If symptoms worsen or fail to improve with rest, it is important to seek medical attention as pneumonia can lead to serious complications.
"""
class DiagnosisRequest(BaseModel):
    report_data: Optional[str]


def generate_diagnosis(report_data: str, top_n_results: List[dict]):
    retrieved_data = "\n\n".join(
        [
            # f"Option {i+1}:\n"
            f"**Disease Name**: {result['Disease name']}\n"
            f"**Symptoms**: {result['Symptoms']}\n"
            f"**Cure**: {result['Cure']}\n"
            f"**Causes**: {result['Causes']}\n"
            f"**Precautions**: {result['Precautions']}\n"
            f"**Medicines**: {result['Medicines']}"
            for i, result in enumerate(top_n_results)
        ]
    )
    prompt = ChatPromptTemplate.from_template(DIAGNOSIS_TEMPLATE)
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"retrieved_data": retrieved_data, "report_data": report_data})
    return {"response": res}


def get_disease_details(disease_name):
    prompt = ChatPromptTemplate.from_template(SYMPTOMS_TEMPLATE)
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"disease_name": disease_name})
    return {"response": res}


def generate_diagnosis_report_for_image(disease_label : str, retrieved_data: str, user_input: str= ""):
    prompt = ChatPromptTemplate.from_template(IMAGE_DIAGNOSIS_TEMPLATE)
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"disease_label": disease_label, "retrieved_data": retrieved_data, "user_input": user_input})
    return {"response": res}


@app.post("/diagnose_text")
async def diagnose(request: DiagnosisRequest):
    report_data = request.report_data
    print("report_data:: ", report_data)
    top_3_results = find_most_similar_sentence(report_data)
    diagnosis_response = generate_diagnosis(report_data, top_3_results)

    return diagnosis_response


@app.post("/diagnose_image")
async def diagnose(file: UploadFile):
    image_content = await file.read()
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as file:
        file.write(image_content)
    print("Filename:: ", file_path)
    predicted_classes = classify_image(file_path)
    if predicted_classes:
        predicted_disease = ", ".join(predicted_classes)
    else:
        predicted_disease = "Cardiomegaly"

    disease_details = get_disease_details(predicted_disease)
    print("disease_details:: ", disease_details)
    diagnosis_response = generate_diagnosis_report_for_image(predicted_disease, disease_details, "")
    return diagnosis_response


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()


def classify_image(image_path: str):
    label_list = [
        "Cardiomegaly",
        "No Finding",
        "Hernia",
        "Infiltration",
        "Nodule",
        "Emphysema",
        "Effusion",
        "Atelectasis",
        "Pleural_Thickening",
        "Pneumothorax",
        "Mass",
        "Fibrosis",
        "Consolidation",
        "Edema",
        "Pneumonia",
    ]

    # Define your transforms for preprocessing the input image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess an image for inference
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = image_model(image)
        probabilities = torch.sigmoid(outputs)
        predicted_labels = (probabilities > 0.5).float()
    print("predicted_labels:: ", predicted_labels)
    predicted_classes = [
        label_list[i]
        for i in range(len(predicted_labels[0]))
        if predicted_labels[0][i] == 1
    ]
    return predicted_classes
