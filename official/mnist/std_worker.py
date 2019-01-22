import tensorflow as tf
import os
import json
from tensorflow.contrib import distribute

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
tf.logging.set_verbosity(tf.logging.INFO)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['10.0.22.3:5000','10.0.24.3:5000'],
        #'ps': ['10.0.22.3:5001','10.0.24.3:5001'],
    },
    'task': {
        'type': 'worker',
        'index': 0,
    },
    'rpc_layer': 'grpc',
})

distribute.run_standard_tensorflow_server().join()

