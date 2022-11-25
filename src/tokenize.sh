# 中文版 資料前處理
# export CLASSPATH=/workplace/yhcheng/PreSumm/standford-corenlp/stanford-corenlp-4.5.1/stanford-corenlp-4.5.1-models-chinese.jar
JSON_PATH=/workplace/yhcheng/PreSumm/json_data/cnewsum
BERT_DATA_PATH=/workplace/yhcheng/PreSumm/bert_data/cnewsum
LOG_FILE=/workplace/yhcheng/PreSumm/logs/CNewSum_data_preprocessing.log
python preprocess.py -language chinese \
    -mode format_to_chinese_bert \
    -min_src_nsents 1 \
    -min_src_ntokens_per_sent 3 \
    -raw_path $JSON_PATH \
    -save_path $BERT_DATA_PATH  \
    -lower True \
    -n_cpus 6 \
    -log_file $LOG_FILE \
