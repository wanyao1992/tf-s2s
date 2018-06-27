## Introduciton
A simple version of seq2seq with tensorflow.

## Features
- Data preprocess
- Attention
- Beam Search

## Run
### Dataset
- src-train.txt
- tgt-train.txt
- src-val.txt
- tgt-val.txt
- src-test.txt
- tgt-test.txt

each of them looks like this:

1. It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
2. The name of this site , and program name Title purchased will not be displayed .

### Preprocessing
```
python preprocess.py -train_src /media/BACKUP/ghproj_d/tf-seq2seq/src-train.txt -train_tgt /media/BACKUP/ghproj_d/tf-seq2seq/tgt-train.txt -valid_src /media/BACKUP/ghproj_d/tf-seq2seq/src-val.txt -valid_tgt /media/BACKUP/ghproj_d/tf-seq2seq/tgt-val.txt -test_src /media/BACKUP/ghproj_d/tf-seq2seq/src-val.txt -test_tgt /media/BACKUP/ghproj_d/tf-seq2seq/tgt-val.txt -save_data /media/BACKUP/ghproj_d/tf-seq2seq/ > ~/log/tf-s2s/log.preprocess
```
### Training
```
python main.py --cell_type 'lstm' --attention_type 'luong' --hidden_units 1024 --depth 2 --embedding_size 500  --mode train >~/log/tf-s2s/log.main.train
```
### Testing
```

```
