# Health Gen AI ChatBot Service

This FastAPI service provides AI-based health diagnostics through both text and image inputs.

## Features

- Disease diagnosis based on patient-provided text
- Disease diagnosis based on patient-provided images

## Prerequisites

- Python 3.7+

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/subhamyadav580/health-gen-ai.git
cd health-gen-ai
```

### 2. Set up a virtual environment
```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the service
```bash
uvicorn main:app --reload --port 8000
```

### 5. Test the service
```bash
http://127.0.0.1:8000/
```