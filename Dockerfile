# Base image with Python
FROM python:3.13

# Arguments passed at build time
ARG DIR_NAME=/workspace
ARG LABEL_NAME=hazard-category
ARG LLM_NAME=llama3.1:8b

# Environment variables
ENV DIR_NAME=${DIR_NAME}
ENV LABEL_NAME=${LABEL_NAME}
ENV LLM_NAME=${LLM_NAME}

# Install Python requirements
RUN pip install numpy pandas scikit-learn==1.6.1 crepes==0.8.0 ollama tqdm requests

# Set working directory
WORKDIR ${DIR_NAME}

# Copy trained base classifier:
COPY python/training/${LABEL_NAME}-lr.pkl cicle.pkl

# Copy prediction files (assume they’re in a subdir named "python/inference" relative to Dockerfile)
COPY python/inference/* .

# Set command
CMD python predict.py ${LLM_NAME} ${LABEL_NAME}