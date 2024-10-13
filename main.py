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


os.environ["OPENAI_API_KEY"] = ("*****")

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "uploaded_images/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists


# model = OllamaLLM(model="llama3.1")
model = ChatOpenAI(model="gpt-3.5-turbo")


model = CNNModel()
model.load_state_dict(
    torch.load(
        "/Users/shubhamyadav/Documents/Personal/chest_xray_cnn_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)  # Load the model state
model.eval()


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

### Example Response:
- **Diagnosis**: Based on the symptoms of fever, cough, and shortness of breath, the most likely diagnosis is pneumonia, as mentioned in Option 1. However, bronchitis and asthma should also be considered if further symptoms arise.
- **Key Insights**: The patient's symptoms align closely with bacterial pneumonia, typically treated with antibiotics. Asthma may also cause shortness of breath, but this is less likely given the absence of wheezing.
- **Potential Symptoms**: Fever, cough with phlegm, and difficulty breathing are commonly associated with pneumonia. Asthma might present with wheezing, which is not reported in the user input.
- **Recommended Next Steps**: It would be advisable for the patient to undergo a chest X-ray and a sputum culture to confirm the diagnosis of pneumonia. If asthma or bronchitis is suspected, spirometry tests could be useful.
- **Relevant Medications**: Amoxicillin or macrolides (such as azithromycin) are often prescribed for bacterial pneumonia. However, a medical consultation is recommended to confirm the diagnosis before prescribing.
"""


class DiagnosisRequest(BaseModel):
    report_data: Optional[str]


def generate_diagnosis(report_data: str, top_n_results: List[dict]):
    retrieved_data = "\n\n".join(
        [
            f"Option {i+1}:\n"
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
        predict_disease = ", ".join(predicted_classes)
    else:
        predict_disease = "chest xray report"
    print("report_data:: ", predict_disease)
    top_3_results = find_most_similar_sentence(predict_disease)
    diagnosis_response = generate_diagnosis(predict_disease, top_3_results)
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
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted_labels = (probabilities > 0.5).float()
    predicted_classes = [label_list[i] for i in range(len(predicted_labels[0])) if predicted_labels[0][i] == 1]
    return predicted_classes


