FROM python:3.10

WORKDIR /app

COPY backend /app

RUN ls -la

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 7860

CMD ["python", "main.py"]