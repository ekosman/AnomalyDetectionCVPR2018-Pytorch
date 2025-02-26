FROM continuumio/miniconda3

# Set up working directory
WORKDIR /app

# Install gdown for Google Drive downloads
RUN pip install gdown

# Clone the repository
RUN git clone https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch.git

RUN conda create -n anomaly python=3.9 -y

ENV PATH=/opt/conda/envs/anomaly/bin:$PATH

# Change to the repo directory
WORKDIR /app/AnomalyDetectionCVPR2018-Pytorch

# Download requirements.txt from Google Drive
RUN gdown --id 12MRBC9yqbG5-YsPgYKAcA6nFzDjwglhd -O requirements.txt

RUN pip install -r requirements.txt

# Download pretrained weights
RUN cd pretrained && \
    wget http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle && \
    cd ..

# Download example videos from Google Drive
RUN gdown --folder --id 14sBuNXOKLa8Jos8S3VCeAgbHOCgF2-Y3 -O example_videos


# Copy setup script to container
COPY setup_anomaly.sh /app/AnomalyDetectionCVPR2018-Pytorch/setup_anomaly.sh

# Make sure the script is executable
RUN chmod +x /app/AnomalyDetectionCVPR2018-Pytorch/setup_anomaly.sh


RUN echo "source activate anomaly" >> ~/.bashrc


# Set the default command to start a bash shell
CMD ["/bin/bash"]
