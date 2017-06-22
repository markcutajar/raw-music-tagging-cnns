gcloud ml-engine local train --package-path trainer \
--module-name trainer.task \
-- \
--train-files $TRAIN_FILE \
--eval-files $EVAL_FILE \
--job-dir $JOB_DIR \
--metadata-files $METDATA_FILE \
--train-steps $TRAIN_STEPS \
--selective-tags $SELTAGS \
--model-function $MODEL
