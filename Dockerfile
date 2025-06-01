FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /
COPY . .
RUN pip install -r requirements.txt
RUN pip install "huggingface_hub[cli]"
# Create model directory
RUN mkdir -p /runpod-volume/

# Download the large model (this is the time-consuming step)
RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /runpod-volume/Wan2.1-I2V-14B-480P

COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]