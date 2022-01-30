# kmeans_server

### General
This server serves a basic page that allows clustering number images that exists in MNist dataset -  http://yann.lecun.com/exdb/mnist/   

The clustering is performed by the KMeans algorithm, using the scikit-learn sdk

###How To Use It
#### From Docker
`docker pull mppeled/kmeans_server:1.0.0`  
`docker run -p 8008:8008 -t mppeled/kmeans_server:1.0.0`  

browse to http://localhost:8008/ and select the clusters amount

#### Running the python server  
`clone https://github.com/morpel/kmeans_server.git`
`pip install -r requirements.txt`
`cd kmeans_server/webapp`
`python webapp.py` (requires python 3.7)

browse to http://localhost:8008/ and select the clusters amount
