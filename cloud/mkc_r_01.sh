TEST_SCRIPT_NAME=mkc_r_01
MODEL=mkc_r

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=gs://magnatagatune_dataset/out_$JOB_NAME
TRAIN_FILE=gs://magnatagatune_dataset/train_rawdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_rawdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/raw_metadata.json
TRAIN_STEPS=13577
LEARNING_RATE=0.1
REGION=us-east1
CONFIG=config.yaml

gcloud ml-engine jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.2 \
--job-dir $JOB_DIR \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
--config $CONFIG \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--train-steps $TRAIN_STEPS \
--metadata-files $METADATA_FILE \
--learning-rate $LEARNING_RATE \
--model-function $MODEL