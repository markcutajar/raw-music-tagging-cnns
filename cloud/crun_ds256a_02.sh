current_date=$(date +%m%d_%H%M%S)
MODEL=ds256a_t50
JOB_NAME=ds256a_02_$current_date
JOB_DIR=gs://magnatagatune_dataset/out_$JOB_NAME
TRAIN_FILE=gs://magnatagatune_dataset/train_rawdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_rawdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/raw_metadata.json
TRAIN_STEPS=22000
REGION=us-east1
CONFIG=config.yaml
WINDOW_SIZE=51776

gcloud ml-engine jobs submit training $JOB_NAME \
--stream-logs \
--runtime-version 1.0 \
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
--model-function $MODEL \
--window-size $WINDOW_SIZE
