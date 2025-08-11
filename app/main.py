import os
import time
import pandas as pd
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

from nlp.gpt_utils import order_files
from extraction.file_utils import get_file_type, clear_output, split_text, move_file
from nlp.topic_modelling import topic_model
from tasks import (
    handle_image_task,
    handle_pdf_task,
    handle_video_task,
    create_outline_task,
    line_sort_task,
    embed_sections_task,
    create_notes_task,
)
from celery import group

from config import NOTE_INPUTS_DIR, RAW_TEXT, SECTIONS, COMPLETED_NOTES_FILE, PREVIOUS_INPUTS

from app import socketio, app   
from threading import Thread
import threading

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Run App
def run_flask():
    print("Flask server starting on http://127.0.0.1:5000")
    app.run(debug=False, use_reloader=False)  # Starts Flask but doesn't block main.py
# Start Flask in a separate thread
flask_thread = Thread(target=run_flask)
flask_thread.start()

# Timer
stop_timer = threading.Event() # Stop event
def timer():
    start_time = time.time()
    while not stop_timer.is_set():
        elapsed = time.time() - start_time
        print(f"\rElapsed time: {elapsed:.2f} seconds")
        socketio.emit("timer", {"time" : elapsed})
        time.sleep(1)

# -----------------------------------------------
#  Main
# -----------------------------------------------

def main():

    clear_output(SECTIONS) # Clear old files 
    move_file(NOTE_INPUTS_DIR, PREVIOUS_INPUTS) # Move old inputs

    # Uploaded Files
    folder = NOTE_INPUTS_DIR
    raw_files = [
        f for f in os.listdir(folder)
        if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))
    ]
    files = sorted(
        raw_files,
        key=lambda f: int(re.search(r"\d+", f).group()) if re.search(r"\d+", f) else 0
    )
    if not files:
        print("No available files in the folder.")
        while not files:
            print("Waiting for files.")
            time.sleep(3)
            files = [
                f for f in sorted(os.listdir(folder),
                key=lambda f: int(re.search(r"\d+", f).group()) if re.search(r"\d+", f) else 0)
                if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))]
    else:
        print("Files found:", files)
    print("")

    open(RAW_TEXT, 'w', encoding="utf-8") # Open raw text file

    # Start the timer thread (starting processing)
    timer_thread = threading.Thread(target=timer)
    timer_thread.start()

    # Work with the available files
    time.sleep(3)
    processing_tasks = []
    for file in files:
        file_path, file_type = get_file_type(file)
        valid_video_types = {"MP4", "AVI", "MKV", "MOV", "WMV", "FLV", "WEBM", "MPEG", "MPG", "OGV", "3GP", "MTS"}
        valid_image_types = {"PNG", "JPEG", "JPG", "BMP", "GIF", "TIFF", "WEBP"}
        try:
            if file_type in valid_image_types:
                processing_tasks.append(handle_image_task.s(file, file_path))
            elif file_type == 'PDF':
                processing_tasks.append(handle_pdf_task.s(file, file_path))
            elif file_type in valid_video_types:
                processing_tasks.append(handle_video_task.s(file, file_path))
        except Exception as e:
            print(f"Invalid file {e}")

    if processing_tasks:
        group(processing_tasks).apply_async().get()

    time.sleep(3)

    # Create an outline from the raw text and return a df with info
    df_records = create_outline_task.delay(RAW_TEXT).get()
    df = pd.DataFrame(df_records)

    section_embeddings = {
        row["filename"]: row["embedding"]
        for row in df_records
    }

    # Sort lines into their respective sections
    line_sort_task.delay(RAW_TEXT, section_embeddings).get()

    # Embed the new sorted sections
    embed_sections_task.delay(SECTIONS).get()

    stop_timer.set()    # Signal the timer thread to stop (processing done)
    timer_thread.join()    # Wait for the timer thread to finish

    print("--------------")
    print("Creating Notes")
    print("--------------")

    # Get the sections
    sections = sorted(
    [f for f in os.listdir(SECTIONS) if not f.startswith('.')],
    key=lambda f: int(re.match(r'\d+', f).group())
    )

    # Create notes on chunks
    note_tasks = [create_notes_task.s(section, df_records) for section in sections]
    if note_tasks:
        group(note_tasks).apply_async().get()

    print("--------------")
    print("Notes Completed")
    print("--------------")


if __name__ == "__main__":
    main()
