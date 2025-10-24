# Qwen LLM + Neo4j K-Pop Music Service

A FastAPI-based backend service that combines **Qwen2.5-7B-Instruct LLM** (for natural language interaction), **Neo4j graph database** (for structured K-Pop knowledge retrieval), and **SerpAPI** (for real-time internet search) to provide intelligent K-Pop music recommendations and information queries.


## Table of Contents
- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)



## Project Overview
This service acts as a backend for K-Pop music assistants, supporting three core capabilities:
1. **Basic LLM Interaction**: Respond to general K-Pop questions using the Qwen2.5-7B model.
2. **Knowledge Base Retrieval**: Fetch structured K-Pop data (e.g., artist relationships, top tracks) from Neo4j when triggered.
3. **Real-Time Search**: Call SerpAPI to get up-to-date K-Pop information (e.g., latest album releases, concert news).

The service exposes a RESTful API for frontend integration and supports CORS for cross-domain access.


## Core Features
| Feature | Description | Trigger Condition |
|---------|-------------|-------------------|
| Basic LLM Response | Answer general K-Pop questions (e.g., "What is K-Pop?") using the pre-trained Qwen model. | No special keywords required. |
| Neo4j Knowledge Base | Retrieve structured data (artist names, top tracks, album links) and inject it into the LLM prompt for accurate recommendations. | User input contains "knowledge base". |
| SerpAPI Internet Search | Fetch real-time data (e.g., "latest BTS album 2024") via SerpAPI and generate answers based on search results. | User input contains "search" or "internet". |
| Singer Detection | Automatically identify K-Pop singer names from user input to target Neo4j queries (e.g., "Tell me about ITZY from knowledge base"). | Triggered with "knowledge base" keyword. |
| CORS Support | Allow cross-origin requests from frontends (configurable for production). | Enabled by default for all origins (dev-only). |


## Tech Stack
| Category | Tools/Libraries | Version Requirement |
|----------|-----------------|---------------------|
| Web Framework | FastAPI | >=0.100.0 |
| LLM | Hugging Face Transformers, Qwen2.5-7B-Instruct | Transformers >=4.35.0 |
| Quantization | BitsAndBytes | >=0.41.1 |
| Database | Neo4j | >=5.0 (with AuraDB support) |
| Search | SerpAPI | SerpAPI Python Wrapper >=0.1.16 |
| Type Hints | Pydantic | >=2.0 |
| Server | Uvicorn | >=0.23.2 |
| Dependencies | PyTorch | >=2.0 (with CUDA support recommended) |


## Prerequisites
Before starting the service, ensure you have the following:
1. **Hardware Requirements**:
   - Minimum: 16GB RAM (for 4-bit quantized Qwen2.5-7B model).
   - Recommended: NVIDIA GPU with ≥8GB VRAM (e.g., RTX 3090, A10) for faster inference.
2. **Software Requirements**:
   - Python 3.9–3.11 (Python 3.12 may have compatibility issues with some libraries).
   - CUDA Toolkit 11.7+ (if using GPU acceleration).
3. **API Keys & Credentials**:
   - SerpAPI Key: Sign up at [SerpAPI](https://serpapi.com/) to get a free/paid key.
   - Neo4j Credentials: A Neo4j database (local or AuraDB) with K-Pop artist data (see [Neo4j Data Preparation](#neo4j-data-preparation)).
4. **Singer List File**: A `singer.txt` file containing K-Pop singer/group names (one per line) for singer detection.


## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/qwen-neo4j-kpop-service.git
cd qwen-neo4j-kpop-service
```

### 2. Create a Virtual Environment
```bash

conda create -n kpop python==3.10

conda activate kpop
```

### 3. Install Dependencies
Install required packages via `pip`:
```bash
pip install -r requirements.txt
```



## Usage Guide

### 1. Start the Service
Run the Uvicorn server (single worker for debugging):
```bash
python main.py  # If renamed from flask_test.py
# Or directly with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 7899 --workers 1 --reload

double click the html file to start
```




