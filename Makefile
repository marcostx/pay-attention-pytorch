# Docker Makefile
PROGRAM="TFN-violence"

CPU_REGISTRY_URL=rwightman
GPU_REGISTRY_URL=rwightman
CPU_VERSION=latest-cpu
GPU_VERSION=latest-gpu
CPU_DOCKER_IMAGE=pytorch-opencv
GPU_DOCKER_IMAGE=pytorch-opencv
APP_NAME=marcostx:gpu

# enable/disable GPU usage
GPU=false
# Config file used to experiment
CONFIG_FILE="configs/config.json"
# List of cuda devises
CUDA_VISIBLE_DEVICES=0
# Name of dataset to process
PROCESS_DATASET=""

#Path to src folder
HOST_CPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_SOURCE_PATH=$(shell pwd)
HOST_GPU_DATASET_PATH=/home/marcostx/master-degree/pay-attention-pytorch/data/
HOST_CPU_DATASET_PATH=/Users/marcostexeira/mediaeval2015_subset/

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker

#HOST VARS
HOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26007

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_SOURCE_PATH=/work/src
IMAGE_METADATA_PATH=$(IMAGE_SOURCE_PATH)/metadata
IMAGE_DATASET_PATH=$(IMAGE_SOURCE_PATH)/datasets



# VOLUMES
DOCKER_DISPLAY_ARGS = -e DISPLAY=${HOST_IP}:0 \
                      --volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \


CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
					 --volume=$(HOST_CPU_DATASET_PATH):$(IMAGE_DATASET_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
					 --volume=$(HOST_GPU_DATASET_PATH):$(IMAGE_DATASET_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH)

DOCKER_PORTS = -p $(HOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host --shm-size 8G  $(GPU_DOCKER_VOLUMES) $(APP_NAME)
	DOCKER_RUN_PORT_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host  $(DOCKER_PORTS) $(GPU_DOCKER_VOLUMES) $(APP_NAME)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host --shm-size 8G $(CPU_DOCKER_VOLUMES) $(APP_NAME)
	DOCKER_RUN_PORT_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host  $(DOCKER_PORTS) $(CPU_DOCKER_VOLUMES) $(APP_NAME)
endif

##############################################################################
############################## CODE VARS #####################################
##############################################################################

#COMMANDS
PYTHON_COMMAND=python3
TORCH_COMMAND=th
EXPORT_COMMAND=export
BASH_COMMAND=bash
TENSORBOARD_COMMAND=tensorboard
WGET_COMMAND=wget
MV_COMMAND=mv
MKDIR_COMMAND=mkdir

TRAIN=main.py
CONFIG_FILE=config/config.json
SCRIPT_LUA=util/create_t7.lua

CREATE_H5_FILE=$(PREPROCESSING_FOLDER)/create_h5_files.py
PARSER_FILE=extract_features_mediaeval.py
MOTION_VECTORS=motion_violence.py
FRAMES_FOLDER=dataset/
DATASET_FOLDER=data/raw_frames/violentflow
OUTPUT_FRAMES_VAL_FILES=valid/
MEDIAEVAL_MAIN_FILE=main.lua
BATCH_SIZE=2

##############################################################################
############################ CODE COMMANDS ###################################
##############################################################################

train t: excuda-devise
	@echo "[Train] Trainning..."
	@$(PYTHON_COMMAND) $(TRAIN) --datapath $(DATASET_FOLDER) --trainBatchSize $(BATCH_SIZE)

train-mediaeval tm: excuda-devise
	@echo "[Train] Training Mediaeval..."
	@$(TORCH_COMMAND) $(MEDIAEVAL_MAIN_FILE)  -gpuid $(CUDA_VISIBLE_DEVICES)

tensorboard tb:
	@echo "[Tensorboard] Running Tensorboard"
	@$(TENSORBOARD_COMMAND) --logdir=$(IMAGE_METADATA_PATH) --host 0.0.0.0

excuda-devise ecd:
ifeq ($(GPU), true)
	@echo "\t Using CUDA_VISIBLE_DEVICES: "$(CUDA_VISIBLE_DEVICES)
	@$(EXPORT_COMMAND) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
endif


dataset-train-motion-vectors mv: excuda-devise
	@echo "[Parse training videos Motion Vectors] Parsing videos..."
	@$(PYTHON_COMMAND) $(MOTION_VECTORS) -d $(DATASET_TRAIN_FOLDER)

dataset-test-motion-vectors mt: excuda-devise
	@echo "[Parse test videos Motion Vectors] Parsing videos..."
	@$(PYTHON_COMMAND) $(MOTION_VECTORS) -d $(DATASET_TEST_FOLDER)

dataset-train ds: excuda-devise
	@echo "[Parse training videos] Parsing videos..."
	@$(PYTHON_COMMAND) $(PARSER_FILE) --dataset_folder $(DATASET_TRAIN_FOLDER)

dataset-train-t7 d: excuda-devise
	@echo "[Torch parser] Creating t7 file..."
	@$(TORCH_COMMAND) $(SCRIPT_LUA)

dataset-test ss:
	@echo "[Parse test videos] Parsing videos..."
	@$(PYTHON_COMMAND) $(PARSER_FILE) --dataset_folder $(DATASET_TEST_FOLDER)

##############################################################################
########################### DOCKER COMMANDS ##################################
##############################################################################
run-train rt: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make train CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

run md: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make train-mediaeval CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

run-test rp: docker-print
	@$(DOCKER_RUN_COMMAND) bash;

run-t7 r7: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset-train-t7 CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) ";

run-parser-train-mv tmv: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset-train-motion-vectors CUDA_VISIBLE_DEVICES=-1";


run-parser-test-mv tmv: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset-test-motion-vectors CUDA_VISIBLE_DEVICES=-1";


run-parser-train rd: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset-train CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) ";

run-parser-test rq: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make dataset-test CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) ";

docker-build db:
	@echo "[Docker] Building image"
	@$(DOCKER_BUILD_COMMAND) ;


#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	@echo "[GPU Docker] Running gpu docker image..."
else
	@echo "[CPU Docker] Running cpu docker image..."
endif
