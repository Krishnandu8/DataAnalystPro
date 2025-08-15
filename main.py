from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging
from fastapi.responses import HTMLResponse
import difflib

from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<p>Frontend file not found.</p>", status_code=404)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def last_n_words(s, n=100):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

@app.post("/api")
async def analyze(request: Request):
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Step-1: Folder created: %s", request_folder)

    form = await request.form()
    question_text = None
    saved_files = {}

    for field_name, value in form.items():
        file_path = os.path.join(request_folder, field_name)
        async with aiofiles.open(file_path, "wb") as f:
            content = await value.read()
            await f.write(content)
        saved_files[field_name] = file_path

        if "questions.txt" in field_name:
            question_text = content.decode('utf-8')

    if not question_text:
        return JSONResponse(status_code=400, content={"message": "questions.txt is a required field."})

    logger.info("Step-2: Files received and saved.")

    # Get code from LLM
    llm_response = None
    try:
        llm_response = await parse_question_with_llm(
            question_text=question_text,
            uploaded_files=list(saved_files.keys()),
            folder=request_folder,
            session_id=request_id
        )
    except Exception as e:
        logger.error("Error getting initial code from LLM: %s", str(e))
        return JSONResponse(status_code=500, content={"message": f"LLM Error: {str(e)}"})
    
    logger.info("Step-3: Received scraping code from LLM.")

    # Execute scraping code
    execution_result = await run_python_code(llm_response["code"], llm_response["libraries"], folder=request_folder)
    
    if execution_result["code"] == 0:
        logger.error("Error executing scraping code: %s", execution_result["output"])
        return JSONResponse(status_code=500, content={"message": "Failed to execute data scraping code.", "details": execution_result["output"]})

    logger.info("Step-4: Scraping code executed successfully.")

    # Get answer code from LLM
    answer_code_response = None
    try:
        answer_code_response = await answer_with_data(
            question_text=llm_response["questions"], 
            folder=request_folder, 
            session_id=request_id
        )
    except Exception as e:
        logger.error("Error getting answer code from LLM: %s", str(e))
        return JSONResponse(status_code=500, content={"message": f"LLM Error during answer generation: {str(e)}"})
    
    logger.info("Step-5: Received answer code from LLM.")

    # Execute answer code
    final_result = await run_python_code(answer_code_response["code"], answer_code_response["libraries"], folder=request_folder)

    if final_result["code"] == 0:
        logger.error("Error executing final answer code: %s", final_result["output"])
        return JSONResponse(status_code=500, content={"message": "Failed to execute final answer code.", "details": final_result["output"]})

    logger.info("Step-6: Final code executed. Reading result.")
    
    result_path = os.path.join(request_folder, "result.json")
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        logger.info("Step-7: Successfully sending result back.")
        return JSONResponse(content=data)
    except Exception as e:
        logger.error("Step-7: Error reading result.json: %s", str(e))
        return JSONResponse(status_code=500, content={"message": "Could not read result file.", "details": str(e)})