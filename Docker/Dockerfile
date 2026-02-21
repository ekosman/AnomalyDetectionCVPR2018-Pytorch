FROM continuumio/miniconda3

# Set up working directory
WORKDIR /app

# Install sudo and gdown for Google Drive downloads
RUN apt-get update && apt-get install -y --no-install-recommends sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install gdown

# Copy repository files from build context (repo root)
COPY . /app/AnomalyDetectionCVPR2018-Pytorch/

RUN conda create -n anomaly python=3.9 -y

ENV PATH=/opt/conda/envs/anomaly/bin:$PATH

# Change to the repo directory
WORKDIR /app/AnomalyDetectionCVPR2018-Pytorch

RUN pip install -r requirements.txt

# Download pretrained weights over HTTPS
RUN mkdir -p pretrained && \
    wget https://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle -O pretrained/c3d.pickle

# Download example videos from Google Drive
RUN gdown --folder --id 14sBuNXOKLa8Jos8S3VCeAgbHOCgF2-Y3 -O example_videos

# Make sure the setup script is executable
RUN chmod +x /app/AnomalyDetectionCVPR2018-Pytorch/Docker/setup_anomaly.sh

# Create non-root user with sudo restricted to apt-get only
RUN useradd -m -s /bin/bash anomalyuser && \
    echo "anomalyuser ALL=(ALL) NOPASSWD: /usr/bin/apt-get" >> /etc/sudoers && \
    chown -R anomalyuser:anomalyuser /app /opt/conda/envs/anomaly

USER anomalyuser

RUN echo "source activate anomaly" >> ~/.bashrc

# Set the default command to start a bash shell
CMD ["/bin/bash"]
