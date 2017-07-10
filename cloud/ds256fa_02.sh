TEST_SCRIPT_NAME=ds256fa_02
MODEL=ds256fa

current_date=$(date +%m%d_%H%M)
JOB_NAME=${TEST_SCRIPT_NAME}_${current_date}
JOB_DIR=gs://magnatagatune_dataset/out_$JOB_NAME
TRAIN_FILE=gs://magnatagatune_dataset/train_win_fbanksdata.tfrecords
EVAL_FILE=gs://magnatagatune_dataset/valid_win_fbanksdata.tfrecords
METADATA_FILE=gs://magnatagatune_dataset/fbank40_win_metadata.json
TRAIN_STEPS=40000
LEARNING_RATE=0.01
EVAL_EPOCHS=1
EVAL_STEPS=106
REGION=us-east1
CONFIG=config.yaml
WINDOWING=STME

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
--eval-steps $EVAL_STEPS \
--eval-num-epochs $EVAL_EPOCHS \
--metadata-files $METADATA_FILE \
--learning-rate $LEARNING_RATE \
--windowing-type $WINDOWING \
--model-function $MODEL