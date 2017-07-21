TEST_SCRIPT_NAME=ds256ra_03
MODEL=ds256rc

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=gs://magnatagatune_dataset/out_$JOB_NAME

TRAIN_FILE=gs://magnatagatune_dataset/train_win_rawdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_win_rawdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/raw_win_metadata.json

TRAIN_STEPS=11000
LEARNING_RATE=0.01
EVAL_STEPS=44
EVAL_BATCH=48
TRAIN_BATCH=80

REGION=us-east1
CONFIG=config.yaml

gcloud ml-engine jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.2 \
--job-dir $JOB_DIR \
--module-name trainer.task_spm \
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
--model-function $MODEL