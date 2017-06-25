current_date=$(date +%Y%m%d_%H%M%S)
MODEL=ds256
JOB_DIR=out_ds256_w322_$current_date
TRAIN_FILE=../magnatagatune/train_rawdata.tfrecords
EVAL_FILE=../magnatagatune/valid_rawdata.tfrecords
METADATA_FILE=../magnatagatune/raw_metadata.json
TRAIN_STEPS=22000
SELTAGS=../magnatagatune/selective_tags.json
WINDOW_SIZE=51776

gcloud ml-engine local train --package-path trainer \
--module-name trainer.task \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--job-dir $JOB_DIR \
--metadata-files $METADATA_FILE \
--train-steps $TRAIN_STEPS \
--model-function $MODEL \
--window-size $WINDOW_SIZE
