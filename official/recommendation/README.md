i### 1. train NeuMF model
#### 1.1 environment requirment
NeuMF model is a tensorflow version of Neural Collaborative Filtering (NCF) framework with Neural Matrix Factorization (NeuMF) model as described in the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper, below are the conditions to be satificated before train this model:
* **TensorFlow**: current version is r1.12
* **Requirments**: there is an text file named `requirments.txt` in `models/official`, this file list all python package requested by NeuMF code, run `sudo pip install -r official/requirements.txt` to auto install python dependencies.
* **Environment Variable**: add path of `official` dir to environment variable `PYTHONPATH` to enable python find requested file by `import official/xxx`  in the begin of code. Assuming that path of ` official ` is `/home/ubuntu/models-master`, the commend is `export PYTHONPATH=$PYTHONPATH:/home/ubuntu/models-master/`.
#### 1.2 command line argument
* **--model_dir**
TensorFlow will save many files during train such as checkpoint and summary files into a fixed dir, and the dir can be specified by `--model_dir` , such as `--model_dir /tmp/ncf_model`.
* **--dataset**
NeuMF model has two dataset:`ml-1m` and `ml-20m`, this argument decide which dataset is used during train.  [Here](https://github.com/tensorflow/models/tree/master/official/recommendation) show the detailed information of the two dataset.
* **--num_gpus**
When train NeuMF model with [distribute strategy](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute), it is import to let distrubute strategy know how many GPU a worker can use, code will pass argument `--num_gpus` to distribute strategy to let it know how many GPU this worker will use, such as `--num_gpus 4`.
* **--clean**
Before train,delete all files in model_dir which is specified by argument `--model_dir`, such as `--clean`.
* **--distribute_strategy**
When distributed train NeuMF models, it is possible to select an distribute strategy, and all value supported now is `ParameterServer` and `Mirror`.  [Here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute)  is the detailed information of each distribute strategy. The argument should be used such as `--distribute_strategy Mirror`
* **--train_epochs**
A epoach is that TensorFlow has finish dealing with all examples in dataset, this argument decide how many epochs code will run, such as `--train_epoachs 1`.
* **--batch_size**
Value of batch size during train, notice that value of this argument is batch size of a process. If train process has 4 GPU, batch size of train process will be `4 * 1024000 `for ParameterServer strategy, `4 * 2048000` for Mirror strategy now. Actually,value `1024000`  is max value of batch size of a GPU when use ParameterServer strategy, value `2048000`  is max value of batch size of a GPU when use Mirror strategy.
* **--cache_id**
Before train NeuMF model, there is a step converting data in csv file into tfrecord file, these tfrecord files save in a dictionary whose name contain a random number such as `1544425230_ncf_recommendation_cache` and tensorFlow will read tfrecord files in this directory during training. When `--cache_id 267304` appreas in command line, name of directory contain a fixed number `267304`, this is useful when train with `Mirror strategy` and will cause an error when use `ParameterServer strategy`. 

There are some other command line argument, and these argument set to default value and no need to change.  Almost all command line argument can be seen in function `define_ncf_flags`  in ` models/official/recommendation/ncf_main.py` , the others are common amoung all other models and can be found in `models/official/utils/flags/_base.py`.
#### 1.3 distributed train
A worker is a train process and train with one worker is called `local train` and train with one more workers is called `distributed train`. 
* **local train**: run `python models/official/recommendation/ncf_main.py` with command line argument will run local train. Notice `--distribute_strategy`  is also can be used in local train.
* **distributed train**:when distributed train, environment variable `TF_CONFIG` is necessary for each worker, `TF_CONFIG` variable contains tasks and ip port pair of each worker. Detailed information about `distributed train` and `TF_COINFIG` see [Distribute TensorFlow](https://www.tensorflow.org/deploy/distributed).

There are two simple scripts to show how to run `local train` and `distributed train`:
* local train:
```
#!/usr/bin/env bash

# dataset and batch size
NUM_GPU=4
DATASET="ml-20m"
BATCH_SIZE_BASE=1024000
export CUDA_VISIBLE_DEVICES="0,1,2,3"
DISTRIBUTE_STRATEGY='ParameterServer'
BATCH_SIZE=`echo $(($NUM_GPU*$BATCH_SIZE_BASE))`
echo "BATCH SIZE IS ${BATCH_SIZE}"

# model save dir
MODEL_DIR='/tmp/ncf_model'
if [ ! -d "${MODEL_DIR}" ]; then
    mkdir ${MODEL_DIR}
else
    echo "MODLE DIR ${MODEL_DIR} already exist"
fi

# log and output save dir
LOG_BASE="/home/ubuntu/ncf_log/1_node_1_worker"
DATE=`date '+%Y-%m-%d'`
LOG_DIR="${LOG_BASE}/${DATE}"

if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
else
    echo "LOG DIR ${LOG_DIR} already exist"
fi

# log file
PID=$$
DATE=`date '+%H-%M-%S'`
FILE_NAME="ncf_ps_log_${PID}_${DATE}.txt"
LOG_FILE="${LOG_DIR}/${FILE_NAME}"

# PYTHONPATH environment variable
export PYTHONPATH='/home/ubuntu/models-master'

# create log file
touch ${LOG_FILE}
echo "output saves to ${LOG_FILE}"

python ../../official/recommendation/ncf_main.py \
    --model_dir ${MODEL_DIR} \
    --dataset ${DATASET} --hooks "" \
    --num_gpus ${NUM_GPU} \
    --clean \
    --distribute_strategy ${DISTRIBUTE_STRATEGY} \
    --train_epochs 1 \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size 100000 \
    --learning_rate 0.0005 \
    --layers 256,256,128,64 --num_factors 64 \
    --hr_threshold 0.635 \
    --ml_perf 2>&1 | tee ${LOG_FILE}

echo "output saves to ${LOG_FILE}"
```
* distributed train:
```
#!/usr/bin/env bash

echo "begin to start NeuMF model train process"

TF_CONFIG='{
    "cluster": {
        "worker": ["10.0.22.2:5000", "10.0.24.3:5000"]
    },
   "task": {"type": "worker", "index": 1},
   "rpc_layer":"grpc"
}'

# dataset and batch size
NUM_GPU=4
DATASET="ml-20m"
BATCH_SIZE_BASE=2048000
export CUDA_VISIBLE_DEVICES="0,1,2,3"
DISTRIBUTE_STRATEGY='Mirror'
BATCH_SIZE=`echo $(($NUM_GPU*$BATCH_SIZE_BASE))`
echo "BATCH SIZE IS ${BATCH_SIZE}"

MODEL_DIR='/tmp/ncf_model'
if [ ! -d "${MODEL_DIR}" ]; then
    mkdir ${MODEL_DIR}
else
    echo "MODLE DIR ${MODEL_DIR} already exist"
fi

# log and output save dir
LOG_BASE="/home/zxy/ncf_log/2_node_2_worker"
DATE=`date '+%Y-%m-%d'`
LOG_DIR="${LOG_BASE}/${DATE}"

if [ ! -d "${LOG_DIR}" ]; then
    mkdir -p ${LOG_DIR}
else
    echo "LOG DIR ${LOG_DIR} already exist"
fi

# log file
PID=$$
DATE=`date '+%H-%M-%S'`
FILE_NAME="ncf_2_node_mirror_log_${PID}_${DATE}.txt"
LOG_FILE="${LOG_DIR}/${FILE_NAME}"

# PYTHONPATH environment variable
export PYTHONPATH='/home/zxy/models-master'

# create log file
touch ${LOG_FILE}
echo "output saves to ${LOG_FILE}"

python ../../official/recommendation/ncf_main.py \
    --model_dir ${MODEL_DIR} \
    --dataset ${DATASET} --hooks "" \
    --num_gpus ${NUM_GPU} \
    --clean \
    --distribute_strategy ${DISTRIBUTE_STRATEGY} \
    --train_epochs 1 \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size 100000 \
    --learning_rate 0.0005 \
    --layers 256,256,128,64 --num_factors 64 \
    --hr_threshold 0.635 \
    --cache_id 267304 \
    --ml_perf 2>&1 | tee ${LOG_FILE}

echo "output saves to ${LOG_FILE}"
```

The different of `local train` script and `distributed train` script is that `distributed train` need `TF_CONFIG` variable and `local train` don't. And when runing `distributed train`, one more worker is needed so run scripts mulitiple times in single server or one more server with `index`   and `type` value changed in `TF_CONFIG`,  value of `rpc_layer`  is `grpc` or `grpc+gdr`  normally, and value of `rpc_layer` decide which transport method want to use. Detailed information of `TF_CONFIG` see [TF_CONFIG environment variable](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute/#tf_config-environment-variable).  Environment variable `CUDA_VISIBLE_DEVICES`  can control how many GPU a worker can use, more information see [here](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/). Notice if `CUDA_VISIBLE_DEVICES` changed such as change from `'0,1,2,3'` to `'0'`, command line argument `--num_gpus` may need to change otherwise distributed strategy will not work well.

### 2. code change
#### 2.1 choose distribute strategy
The function `construct_estimator`  in `ncf_main.py` create estimator with distribute strategy, and original version choose distribute strategy with number of GPU a worker can use:
```
  distribution = distribution_utils.get_distribution_strategy(num_gpus=num_gpus)
  run_config = tf.estimator.RunConfig(train_distribute=distribution,eval_distribute=distribution)
  params["eval_batch_size"] = eval_batch_size
  model_fn = neumf_model.neumf_model_fn
  if params["use_xla_for_gpu"]:
    tf.logging.info("Using XLA for GPU for training and evaluation.")
    model_fn = xla.estimator_model_fn(model_fn)
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                     config=run_config, params=params)
  return estimator, estimator
```

new version choose distribute strategy via command line argument:
```
  tf.logging.info("num_gpus is %d in construct_estimator" % num_gpus)
  distribute_strategy = params['distribute_strategy']
  if distribute_strategy == 'ParameterServer':
    distribution = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=num_gpus)
    tf.logging.info("num_gpus_per_worker is %d in ParameterServerStrategy" % num_gpus)
    tf.logging.info("num tower of ParameterStrategy is %d" % distribution.num_towers)
    tf.logging.info("distribute strategy is ParameterServer")
  elif distribute_strategy == 'Mirror':
    tf.logging.info("distribute strategy is Mirror")
    distribution = distribution_utils.get_distribution_strategy(num_gpus=num_gpus)
  else:
    tf.logging.info("No distribute strategy found,exit")
    exit(1)

  run_config = tf.estimator.RunConfig(
    # save_checkpoints_steps=None,
    # save_checkpoints_secs=None,
    model_dir=model_dir,
    log_step_count_steps=10,
    train_distribute=distribution,
    eval_distribute=distribution
  )
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                     config=run_config, params=params)
  return estimator
```
#### 2.2 change API to support distribute train
Origin version use `tf.estimator.Estimator.trian`  to train and use `tf.estimator.Estimator.evaluate` to evaluate in `run_ncf` function in `ncf_main.py`, and it is hard to train in multiple workers,and new version use `tf.estimator.train_and_evaluate`  in `run_ncf` function to run local and distributed train.

original version:
```
train_estimator.train(input_fn=train_input_fn, hooks=train_hooks,steps=num_train_steps)
eval_results = eval_estimator.evaluate(eval_input_fn,steps=num_eval_steps)
```

new version:
```
   train_spec = tf.estimator.TrainSpec(train_input_fn,max_steps=5000,hooks=train_hooks)
   eval_spec = tf.estimator.EvalSpec(eval_input_fn,steps=100)
   tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
```
### 3. official README.md
This file is not official's original README.md, official's README.md see [here](https://github.com/tensorflow/models/blob/master/official/recommendation/README.md).
