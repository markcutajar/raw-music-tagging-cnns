current_date=$(date +%m%d_%H%M%S)
MODEL=basic
JOB_DIR=out_basic_$current_date
TRAIN_FILE=../magnatagatune/train_rawdata.tfrecords
EVAL_FILE=../magnatagatune/valid_rawdata.tfrecords
METADATA_FILE=../magnatagatune/raw_metadata.json
TRAIN_STEPS=18102
LEARNING_RATE=0.1
NUM_SAMPLES=3000

gcloud ml-engine local train --package-path trainer \
--module-name trainer.task \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--job-dir $JOB_DIR \
--metadata-files $METADATA_FILE \
--train-steps $TRAIN_STEPS \
--learning-rate $LEARNING_RATE \
--num-song-samples $NUM_SAMPLES \
--model-function $MODEL
