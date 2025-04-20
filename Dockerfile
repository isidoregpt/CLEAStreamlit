FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repo
COPY . .

EXPOSE 8501

# <-- point to your actual Streamlit script!
ENTRYPOINT ["streamlit", "run", "CLEAStreamlit.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
