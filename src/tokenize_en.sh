# 中文版 資料前處理
# export CLASSPATH=/workplace/yhcheng/PreSumm/standford-corenlp/stanford-corenlp-4.5.1/stanford-corenlp-4.5.1-models-chinese.jar
JSON_PATH=/workplace/yhcheng/PreSumm/json_data/cnndm
BERT_DATA_PATH=/workplace/yhcheng/PreSumm/bert_data
LOG_FILE=/workplace/yhcheng/PreSumm/logs/cnntest.log
python preprocess.py -language english -mode format_to_bert -raw_path $JSON_PATH -save_path $BERT_DATA_PATH  -lower -n_cpus 1 -log_file $LOG_FILE