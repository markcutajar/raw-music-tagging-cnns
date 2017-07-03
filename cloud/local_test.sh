TEST_SCRIPT_NAME=cloud_test
MODEL=mkc_r

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=out_${JOB_NAME}
TRAIN_FILE=../magnatagatune/train_rawdata.tfrecords
EVAL_FILE=../magnatagatune/valid_rawdata.tfrecords
METADATA_FILE=../magnatagatune/raw_metadata.json

TRAIN_STEPS=10
LEARNING_RATE=0.1

gcloud ml-engine local train --package-path trainer \
--module-name trainer.task \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--job-dir $JOB_DIR \
--metadata-files $METADATA_FILE \
--train-steps $TRAIN_STEPS \
--learning-rate $LEARNING_RATE \
--model-function $MODEL