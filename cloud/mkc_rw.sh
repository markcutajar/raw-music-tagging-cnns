TEST_SCRIPT_NAME=mkc_rw_01
MODEL=mkc_rw

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=gs://magnatagatune_dataset/out_$JOB_NAME
TRAIN_FILE=gs://magnatagatune_dataset/train_win_rawdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_win_rawdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/raw_win_metadata.json

REGION=us-east1
CONFIG=config.yaml

TRAIN_STEPS=21722
EVAL_STEPS=1272
LEARNING_RATE=0.1


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
--eval-steps $EVAL_STEPS \
--train-steps $TRAIN_STEPS \
--metadata-files $METADATA_FILE \
--learning-rate $LEARNING_RATE \
--model-function $MODEL