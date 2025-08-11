import os
import pandas as pd
from celery import Celery
from extraction.file_utils import handle_image, handle_pdf, handle_video
from extraction.outline import create_outline
from extraction.line_embed import line_sort
from nlp.embedding_utils import embed_sections
from generation.generate_notes import create_notes


celery = Celery(
    'tasks',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)


@celery.task
def handle_image_task(file, file_path):
    return handle_image(file, file_path)


@celery.task
def handle_pdf_task(file, file_path):
    return handle_pdf(file, file_path)


@celery.task
def handle_video_task(file, file_path):
    return handle_video(file, file_path)


@celery.task
def create_outline_task(raw_text_path):
    df = create_outline(raw_text_path)
    return df.to_dict(orient='records')


@celery.task
def line_sort_task(raw_text_path, section_embeddings):
    return line_sort(raw_text_path, section_embeddings)


@celery.task
def embed_sections_task(sections_dir):
    return embed_sections(sections_dir)


@celery.task
def create_notes_task(section, df_records):
    df = pd.DataFrame(df_records)
    return create_notes(section, df)
