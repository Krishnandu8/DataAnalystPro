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

    for field_name in form.keys():
        value = form.get(field_name)
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

    # Get scraping code from LLM
    try:
        llm_response = await parse_question_with_llm(
            question_text=question_text,
            uploaded_files=list(saved_files.keys()),
            folder=request_folder,
            session_id=request_id
        )
        logger.info("Step-3: Received scraping code from LLM.")
    except Exception as e:
        logger.error("Error getting initial code from LLM: %s", str(e))
        return JSONResponse(status_code=500, content={"message": f"LLM Error: {str(e)}"})

    # Execute scraping code
    execution_result = await run_python_code(llm_response["code"], llm_response["libraries"], folder=request_folder)
    
    if execution_result["code"] == 0:
        logger.error("Error executing scraping code: %s", execution_result["output"])
        return JSONResponse(status_code=500, content={"message": "Failed to execute data scraping code.", "details": execution_result["output"]})
    logger.info("Step-4: Scraping code executed successfully.")

    metadata_path = os.path.join(request_folder, "metadata.txt")
    if not os.path.exists(metadata_path):
        error_message = f"Scraping code executed successfully, but failed to create metadata.txt at {metadata_path}."
        logger.error(error_message)
        return JSONResponse(status_code=500, content={"message": error_message})

    # Get answer code from LLM
    try:
        answer_code_response = await answer_with_data(
            question_text=llm_response["questions"], 
            folder=request_folder, 
            session_id=request_id
        )
        logger.info("Step-5: Received answer code from LLM.")
    except Exception as e:
        logger.error("Error getting answer code from LLM: %s", str(e))
        return JSONResponse(status_code=500, content={"message": f"LLM Error during answer generation: {str(e)}"})
    
    # --- START: New Retry Loop ---
    max_attempts = 3
    for attempt in range(max_attempts):
        logger.info(f"Step-6: Executing final code (Attempt {attempt + 1}/{max_attempts}).")
        final_result = await run_python_code(answer_code_response["code"], answer_code_response["libraries"], folder=request_folder)

        if final_result["code"] == 1:
            logger.info("Step-6: Final code executed successfully!")
            break  # Success, exit the loop
        else:
            logger.error(f"Execution failed on attempt {attempt + 1}. Error: {final_result['output']}")
            if attempt < max_attempts - 1:
                logger.info("Asking LLM to fix the code.")
                try:
                    retry_message = last_n_words(final_result['output'])
                    answer_code_response = await answer_with_data(
                        retry_message=retry_message,
                        question_text=llm_response["questions"],
                        folder=request_folder,
                        session_id=request_id
                    )
                    logger.info("Received corrected code from LLM.")
                except Exception as e:
                    logger.error(f"LLM failed to provide a fix: {e}")
                    return JSONResponse(status_code=500, content={"message": "LLM could not fix the code.", "details": str(e)})
            else:
                logger.error("Max retry attempts reached. Could not execute final code.")
                return JSONResponse(status_code=500, content={"message": "Failed to execute final answer code after retries.", "details": final_result["output"]})
    # --- END: New Retry Loop ---

    result_path = os.path.join(request_folder, "result.json")
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        logger.info("Step-7: Successfully sending result back.")
        return JSONResponse(content=data)
    except Exception as e:
        logger.error("Step-7: Error reading result.json: %s", str(e))
        return JSONResponse(status_code=500, content={"message": "Could not read result file.", "details": str(e)})