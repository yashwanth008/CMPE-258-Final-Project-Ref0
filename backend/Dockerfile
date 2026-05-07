# 1. Base Image
FROM python:3.11-slim

# 2. Install modern libraries for MediaPipe/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Create a non-root user (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 4. Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of your backend code
COPY --chown=user . .

# 6. Start the server on port 7860 (Hugging Face default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]