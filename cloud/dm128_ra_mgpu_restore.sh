TEST_SCRIPT_NAME=dm128_ra
MODEL=dm128_ra

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=gs://magnatagatune_dataset/out_dm128_ra_0725_1120

TRAIN_FILE=gs://magnatagatune_dataset/train_win_rawdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_win_rawdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/raw_win_metadata.json

TRAIN_STEPS=11000
LEARNING_RATE=0.1
EVAL_STEPS=44
EVAL_BATCH=12
TRAIN_BATCH=20
WINDOWING=SPM

REGION=us-east1
CONFIG=config.yaml

gcloud ml-engine jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.2 \
--job-dir $JOB_DIR \
--module-name trainer.task_mgpu \
--package-path trainer/ \
--region $REGION \
--config $CONFIG \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps $TRAIN_STEPS \
--eval-steps $EVAL_STEPS \
--eval-batch-size $EVAL_BATCH \
--train-batch-size $TRAIN_BATCH \
--metadata-files $METADATA_FILE \
--learning-rate $LEARNING_RATE \
--windowing-type $WINDOWING \
--model-function $MODEL