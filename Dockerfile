FROM python:3.11-slim
EXPOSE 8080
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# Streamlit needs to run on port 8080 for Cloud Run
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]