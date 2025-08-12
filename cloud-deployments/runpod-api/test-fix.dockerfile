FROM python:3.11-slim

# Quick test of vLLM v0.8.0 vs v0.9.0 compatibility
RUN pip install vllm==0.8.0

# Simple test script
RUN echo 'import vllm; print("vLLM version:", vllm.__version__)' > test.py

CMD ["python", "test.py"]