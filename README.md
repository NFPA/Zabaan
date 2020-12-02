# Language Translation
### Prototype Translation system using Neural Machine Translation (NMT) architectures

<p align="left">
  <img src="logo/NFPA_logo.png" height="80" width="80" title="National Fire Protection Association">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="logo/WPI_Inst_Prim_FulClr.png" height="80" title="Worcester Polytechnic Institute">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  
A prototype translation platform that uses an efficient hybrid approach of human-in-the-loop (HITL) and  Neural Machine Translation (NMT) techniques to suggest domain specific translations to NFPA authors. The system makes use of NFPA datasets like Code & Standards, Research and Outreach material. Developed in collaboration with WPI GQP Program, NFPA International group and Data Analytics team, primarily focused on English-Spanish language pair. The UI front end has options for you to get instant translations and edit incorrect ones suggested by the NMT machine.

<p align="left">
  <img src="logo/NFPA Zabaan gif.gif" height="360" width="826" title="Zabaan - Neural Machine Translation Platform">

## Built With

- [OpenNMT-tf](https://opennmt.net/OpenNMT-tf/) - A general purpose sequence learning toolkit using TensorFlow
- [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) -  A flexible, high-performance serving system for machine learning models
- [Tornado Web Framework](https://www.tornadoweb.org/en/stable/) - A Python web framework and asynchronous networking library
- [MongoDB](https://www.mongodb.com/) - A document-based, distributed database built for modern application developers

## Getting Started

### I. Project Setup

1. Clone the Repository

```python
git clone https://github.com/NFPA/Zabaan.git
cd Zabaan/Serving
```

2. Activate python 3.6 environment (Assuming your using the EC2 Instance with Deep Learning AMI)

```
source activate tensorflow_p36
```

2. Install packages
This installes the python packages for Tokenization, TFServing API 1.X, PyMongo

```
pip install -r requirements.txt
```

3. Get a Latest MongoDB docker image to the machine and map to a directory. Change the volumn path and port number accordingly.

```python
mkdir mongodb
docker run --name gqp-mongo -d -v /home/ubuntu/Zabaan/Serving/mongodb:/data -p 27017:27017 mongo:latest
```

Once you start/stop the MongoDB docker image, for next time just start with container ID/name, no need to download the image again.

```
docker start <container_name/id>
```

4. Copy all the serving models into models folder. You can download the sample model from [here](https://nfpa-translation-models.s3.us-east-2.amazonaws.com/euro_attention.zip)

```python
cp /home/ubuntu/demo/models/* /home/ubuntu/Zabaan/Serving/nfpa_models/
```

We have trained all the models in OpenNMT-tf format. For more details on OpenNMT-tf Saved Model format and Creating/Serving OpenNMT Models. Please see [OpenNMT Serving](http://opennmt.net/OpenNMT-tf/serving.html)

For more details on Serving tensorflow models. Please see Tensorflow Serving

5. Check the [model.config](./Serving/nfpa_models/models.config) so it has the required configuration of the model you want to serve. 

```python
config: {
    name: "name_of_the_model",
    base_path: "/realtive/path/to/model",
    model_platform: "tensorflow"
  }
```

4. With the MongoDB docker started, Start a Tensorflow Serving GPU instance in the background. 
Note: Change the source path accordingly, put your absolute path here. After you start the docker image, you can use to check if success.

```python
nvidia-docker run --name tf_server -d --rm -p 8500:8500 --mount type=bind,source=/home/ubuntu/Zabaan/Serving/nfpa_models/,target=/models/nfpa_models -t tensorflow/serving:1.11.0-gpu --model_config_file=/models/nfpa_models/models.config
```
Verify TF Server started using docker log command:

```python
 docker container logs tf_server
```
It should give you something like below at the end of log file:

```unix
2020-11-16 20:50:35.246427: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: euro_attention version: 1564872567}
2020-11-16 20:50:35.251353: I tensorflow_serving/model_servers/server.cc:285] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2020-11-16 20:50:35.255347: I tensorflow_serving/model_servers/server.cc:301] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 235] RAW: Entering the event loop ...
```

5. Start the server, the mapped endpoints in this file call the requires functions and models from client file.

```python
python server.py --port 8500 --model_name euro_attention
```

6. Application should be running on localhost:8080

## Results 

BLEU Scores on NFPA Data
 
|   | En-Es  | Es-En  |
|---|---|---|
|Before Domain Adaption   | 35.98  | 41.3  |
|After Domain Adaption   | 65.89   | 73.25  |

## Acknowledgments

- [NFPA Data Analytics](https://nfpa.org/News-and-Research/Data-research-and-tools/NFPA-Data-Lab) Team for testing and providing feedback
- WPI [Data Science Graduate Qualifying Project (GQP)](https://www.wpi.edu/academics/departments/data-science/graduate-qualifying-project) Initiative.
