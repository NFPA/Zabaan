# NMT-Pipeline

1. Tokenizing the data
2. Generating word alignments
3. Generating vocabulary
4. Train the Model
5. Domain Adaption
6. Merge Vocab
7. Train the Model for Domain Adaption
8. Inference
9. Detokenization
10. Calculating BLEU Score

## 1. Tokenizing the Data
Firstly, tokenize the dataset. Below, you can find the commands to tokenize your training file which is `train.en` and `train.es` using the configuration file `tokenize_config.yml`.
```
>> cat train.en | onmt-tokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config tokenize_config.yml > tokenized_train.en 

>> cat train.es | onmt-tokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config tokenize_config.yml > tokenized_train.es

tokenize_config.yml : Configuration for tokenization.
tokenized_train.en : saving the output which is in English in a file named tokenized_train.en
tokenized_train.es : saving the output which is in Spanish in a file named tokenized_train.en
```
Similarly, using the above mentioned commands, tokenize your test and eval files.


## 2. Generating Word Alignments
```
>> pr  --merge  --omit-header --join-lines --sep-string=’ ||| ‘ tokenized_train.en tokenized_train.es > align_input.txt
```

Below are the commands to install fast_align
```
>> sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
>> git clone https://github.com/clab/fast_align.git
>> cd fast_align
>> mkdir build
>> cd build
>> cmake ..
>>  make
```
After installing, run the following commands to generate alignment file ie., `corpus_en_es.gdfa`.
```
>> ./fast_align -i align_input.txt -d -o -v > forward_align_en_es.align
>> ./fast_align -i align_input.txt -r -d -o -v > reverse_align_en_es.align
>> ./atools -i  forward_align_en_es.align -j reverse_align_en_es.align -c grow-diag-final-and >  corpus_en_es.gdfa
```

## 3. Generating Vocabulary
Use the following commands to build vocabulary on your tokenized dataset, we can specify the size of the vocabulary using the `--size` option and save the vocabulary in `src-vocab.txt` and `tgt-vocab.txt`.
```
>> onmt-build-vocab --size 80000 --save_vocab src-vocab.txt  tokenized_train.en
>> onmt-build-vocab --size 80000 --save_vocab tgt-vocab.txt  tokenized_train.es
```

## 4.Training the Base model
Here, we are using Transformer model, so we can specify that using `--model_type` option and along with that provide `config.yml` file.
```
>> onmt-main train_and_eval --model_type Transformer --config config.yml --auto_config --num_gpus 8
```

## 5. Domain Adaption
After the base model, on Europarl dataset has been built, we start domain adaption on teh NFPA corpus. To do so, we first have to tokenize the NFPA dataset.

Now we use the same `tokenize_config.yml` file (same as in Step 1) for tokenizing the NFPA dataset and save the corressponding tokenized English and Spanish output in the files `nfpa_tokenize_train.en` and `nfpa_tokenize_train.en` respectively.
```
>> cat NFPA_CS_Train.en | onmt-tokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config tokenize_config.yml > nfpa_tokenize_train.en 
>> cat NFPA_CS_Train.es | onmt-tokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config tokenize_config.yml > nfpa_tokenize_train.es
```
Similary, by using the above commands tokenize the NFPA Test and Eval files as well.

After tokenizing we generate the vocabulary on the tokenized NFPA dataset and save them in the files `nfpa-src-vocab.txt` and `nfpa-tgt-vocab.txt` using the `--save_vocab` option.

```
>> onmt-build-vocab --size 50000 --save_vocab nfpa-src-vocab.txt  nfpa_tokenize_train.en
>> onmt-build-vocab --size 50000 --save_vocab nfpa-tgt-vocab.txt nfpa_tokenize_train.es
```
## 6. Merging the Vocabulary for Domain Adaption
Before starting the domain adaption it is necessary to merge the vocabulary generated from general model and the NFPA vocabulary and have an enriched vocabulary to train the domain adaption on using the `onmt-update-vocab` command.
```
>>onmt-update-vocab --model_dir new_run4/ --output_dir new_run4/update_ckp/ --src_vocab src-vocab.txt --tgt_vocab tgt-vocab.txt --new_src_vocab nfpa-src-vocab.txt --new_tgt_vocab nfpa-tgt-vocab.txt 
```

## 7. Training Model for Domain Adaption

We use the same command and model type to train for domain adaption as well, but here we use a different configuration file which has the merged vocabulary we generated in the previous step.
```
>> onmt-main train_and_eval --model_type Transformer --config config_domain.yml --auto_config --num_gpus 8
```
 
## 8. Inference

After generating the model, we use the test dataset in english to generate a prediction file in spanish which is `nfpa_tokenize_test-op.es` file using the latest checkpoint after domain adaption.

```
>> onmt-main infer --config config_domain.yml --features_file nfpa_tokenize_test.en --predictions_file nfpa_tokenize_test_op.es --checkpoint /new_run4/update_ckp/model.ckpt-15000
```

## 9. Detokenizing the Predictions file
Since, the generated predictions file (See step 8) is tokenized, inorder to calculate the BLEU score with the original test file in spanish needs to be detokenized. Inorder to detokenize, we again use the `tokenize_config.yml` file which was used in the first place to tokenize it.

```
>> cat nfpa_tokenize_test_op.es | onmt-detokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config tokenize_config.yml > nfpa_tokenize_test_op_detoken.es
```
## 10. Calculating the BLEU Score

To calculate the BLEU score, we can use the `multi-bleu-detok.perl` script provided in `OpenNMT-tf/thirdparty/` folder.

```
>> perl multi-bleu-detok.perl NFPA_CS_Test.es < nfpa_tokenize_test_op_detoken.es
```






