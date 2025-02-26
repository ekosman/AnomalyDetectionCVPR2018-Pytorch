# **Anomaly Detection with Docker 🐳**  

This repository provides a **Dockerized** environment for running the **Anomaly Detection CVPR2018** model.  
Follow the steps below to set up, run the container, and make predictions.  

---
## **🚀 Build the Docker Environment**  

Navigate to the **Docker** directory and build the Docker image:  

```bash
cd Docker
sudo docker build -t anomaly .
```

To allow GUI-based visualization, run the following command outside the container:

## **🏃‍♂️ Run the Docker Container**
To allow GUI-based visualization, run the following command outside the container:
```bash
xhost +SI:localuser:$(whoami)
```

Now, start the container:
```bash
sudo docker run --rm -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    anomaly bash
```
Once inside the Docker container, install all dependencies by executing:

```bash
bash setup_anomaly.sh
```
## Run a prediction 🔍
Choose a video from the example_videos directory and run the following command to perform anomaly detection:
```bash
python video_demo.py --feature_extractor "pretrained/c3d.pickle" \
                     --feature_method "c3d" \
                     --ad_model "exps/c3d/models/epoch_80000.pt"
```