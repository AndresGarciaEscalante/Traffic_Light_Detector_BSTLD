import os

# Install the Object Detection 
## Install Tensorflow
os.system('pip install tensorflow-gpu==2.3.1')
os.system('pip3 install --upgrade tensorflow==2.3.1')

## Install Protoc
os.system('curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip')
os.system(unzip protoc-3.13.0-linux-x86_64.zip -d /usr/local bin/protoc)
os.system(unzip protoc-3.13.0-linux-x86_64.zip -d /usr/local include/*)
os.system(rm -f protoc-3.13.0-linux-x86_64.zip)

## Install Tensorflow Api Object Detection 
os.system(git clone https://github.com/tensorflow/models.git)
os.chdir("models/research")
os.system(protoc object_detection/protos/*.proto --python_out=.)
os.system(cp object_detection/packages/tf2/setup.py .)
os.system(python -m pip install --use-feature=2020-resolver .)
os.system(pip install git+https://github.com/google-research/tf-slim)
os.system(pip install tf-models-official)
os.system(pip3 install --upgrade tensorflow==2.3.1)
os.system(CUDA_VISIBLE_DEVICES=0 python object_detection/builders/model_builder_tf2_test.py)

## Open CV dependencies
os.system(apt-get update)
os.system(apt-get install -y libgl1-mesa-dev)