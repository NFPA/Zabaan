from __future__ import print_function

import argparse
import pyonmttok
import pandas as pd
import glob, os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.plotly as py
import plotly.graph_objs as go

import tensorflow as tf

import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

CODE_DIR = 'SampleFilesList'
VOCAB_DIR = 'vocabs'


def pad_batch(batch_tokens):
    """Pads a batch of tokens."""
    lengths = [len(tokens) for tokens in batch_tokens]
    max_length = max(lengths)
    for tokens, length in zip(batch_tokens, lengths):
        if max_length > length:
            tokens += [""] * (max_length - length)
    return batch_tokens, lengths, max_length


def extract_all_prediction(result):
    """Parses a translation result.
    Args:
      result: A `PredictResponse` proto.
    Returns:
      A generator over the hypotheses.
    """
    batch_lengths = tf.make_ndarray(result.outputs["length"])
    batch_predictions = tf.make_ndarray(result.outputs["tokens"])
    for hypotheses, lengths in zip(batch_predictions, batch_lengths):
        for i in range(0, len(lengths)):
            temp_hypothesis = hypotheses[i]
            temp_length = lengths[i] - 1  # Ignore </s>
            yield temp_hypothesis[:temp_length]


def extract_best_prediction(result):
    """Parses a translation result.
    Args:
      result: A `PredictResponse` proto.
    Returns:
      A generator over the hypotheses.
    """
    batch_lengths = tf.make_ndarray(result.outputs["length"])
    batch_predictions = tf.make_ndarray(result.outputs["tokens"])
    # Only consider the first hypothesis (the best one).
    for hypotheses, lengths in zip(batch_predictions, batch_lengths):
        best_hypothesis = hypotheses[0]
        best_length = lengths[0] - 1  # Ignore </s>
        yield best_hypothesis[:best_length]


def extract_best_alignment(result):
    """Parses a alignment result.
    Args:
      result: A `PredictResponse` proto.
    Returns:
      A generator over the hypotheses.
    """
    batch_lengths = tf.make_ndarray(result.outputs["length"])
    batch_predictions = tf.make_ndarray(result.outputs["alignment"])

    best_hypothesis = batch_predictions[0][0]
    best_length = batch_lengths[0][0] - 1 # Ignore </s>
    return best_hypothesis[:best_length]


def send_request(stub, model_name, batch_tokens, timeout=5.0):
    """Sends a translation request.
    Args:
      stub: The prediction service stub.
      model_name: The model to request.
      tokens: A list of tokens.
      timeout: Timeout after this many seconds.
    Returns:
      A future.
    """
    batch_tokens, lengths, max_length = pad_batch(batch_tokens)
    batch_size = len(lengths)
    request = predict_pb2.PredictRequest()
    print("Model Name: ", model_name)
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto(batch_tokens, shape=(batch_size, max_length)))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto(lengths, shape=(batch_size,)))
    return stub.Predict.future(request, timeout)


def translate(stub, model_name, batch_text, tokenizer, timeout=60.0):
    """Translates a batch of sentences.
    Args:
      stub: The prediction service stub.
      model_name: The model to request.
      batch_text: A list of sentences.
      tokenizer: The tokenizer to apply.
      timeout: Timeout after this many seconds.
    Returns:
      A generator over the detokenized predictions.
    """
    batch_input = [tokenizer.tokenize(batch_text.lower())[0]]
    print("Raw Input: ", batch_text)
    print("Processed Input: ", batch_input)
    
    future = send_request(stub, model_name, batch_input, timeout=timeout)
    result = future.result()

    best_output = [prediction for prediction in extract_best_prediction(result)][0]
    best_output = [word.decode('utf-8') for word in best_output]
    alignment = extract_best_alignment(result)

    tokenized_all_output = [tokenizer.detokenize(list(prediction)) for prediction in extract_all_prediction(result)]
    # All translation results provided by model
    output_processed = [sentence.strip("\"") for sentence in tokenized_all_output]

    vocab_check_en = vocab_exist(batch_input[0], language="en", check_tag=True)
    vocab_check_es = vocab_exist(best_output, language="es", check_tag=True)

    fig = alignplot(alignment, batch_input[0], best_output, False)
    print("Best Output: ", output_processed[0])
    
    return output_processed, batch_input, vocab_check_en, fig # vocab_check_es


def translate_multiway(stub, model_name, batch_text, tokenizer, timeout=60.0, tgt_lang='es'):
    """Translates a batch of sentences.
    Args:
      stub: The prediction service stub.
      model_name: The model to request.
      batch_text: A list of sentences.
      tokenizer: The tokenizer to apply.
      timeout: Timeout after this many seconds.
      tgt_lang: Target language
    Returns:
      A generator over the detokenized predictions.
    """
    if (tgt_lang == 'es'):
        lang_tokens = ['__opt_src_en', '__opt_tgt_es']
    elif (tgt_lang == 'fr'):
        lang_tokens = ['__opt_src_en', '__opt_tgt_fr']
    elif (tgt_lang == 'it'):
        lang_tokens = ['__opt_src_en', '__opt_tgt_it']
    elif (tgt_lang == 'pt'):
        lang_tokens = ['__opt_src_en', '__opt_tgt_pt']
    elif (tgt_lang == 'ro'):
        lang_tokens = ['__opt_src_en', '__opt_tgt_ro']

    batch_input = [tokenizer.tokenize(batch_text.lower())[0]]
    batch_input_lang = [[*lang_tokens, *batch_input[0]]]
    print("----------english to ", tgt_lang, "----------")
    print("Raw Input: ", batch_text)
    print("Processed Input: ", batch_input)

    future = send_request(stub, model_name, batch_input_lang, timeout=timeout)
    result = future.result()
    # print("Result(before detok: )"+ str(result))
    # print(str(type(result)))

    best_output = [prediction for prediction in extract_best_prediction(result)][0]
    best_output = [word.decode('utf-8') for word in best_output]
    alignment = extract_best_alignment(result)
    
    # for prediction in extract_all_prediction(result):
    #     print(str(type(prediction)))
    #     print(str(prediction))

    tokenized_all_output = [tokenizer.detokenize(list(prediction)) for prediction in extract_all_prediction(result)]
    # All translation results provided by model
    output_processed = [sentence.strip("\"") for sentence in tokenized_all_output]

    vocab_check_en = vocab_exist(batch_input[0], language="en", check_tag=True)

    fig = alignplot(alignment, batch_input[0], best_output, False)
    print("Best Output: ", output_processed[0])
    return output_processed, batch_input, vocab_check_en, fig


def translate_file(stub, model_name, data, tokenizer, timeout=60.0):
    batch_input = [tokenizer.tokenize(text.lower())[0] for text in data]
    print("Processed Input: ")
    for text in batch_input:
        print(text)
    future = send_request(stub, model_name, batch_input, timeout=timeout)
    result = future.result()
    batch_output = [tokenizer.detokenize(prediction) for prediction in extract_best_prediction(result)]
    return batch_output

def alignplot(align_data, en_tokens = None, es_tokens = None, annot = False):
    """
    plot the align data with tokens in both language
    :params: annot: whether give annot on each element in the matrix
    :params: align_data: attention matrix, array-like
    :params: en_tokens: english tokens (list, array)
    :params: es_tokens: spanish tokens (list, array)
    """
    align_data_shape = align_data.shape
    if en_tokens is not None and es_tokens is not None:
        if annot:
            fig = plt.figure(figsize = (align_data_shape[0]/3,align_data_shape[1]/3))
            sns.heatmap(align_data, cmap = "Reds", annot=annot, fmt=".1f", cbar = True, linewidths=.5, linecolor='gray', xticklabels = en_tokens, yticklabels = es_tokens)
        else:
            fig = plt.figure()
            sns.heatmap(align_data, cmap = "Reds", annot=annot, fmt=".1f", cbar = True, linewidths=.5, xticklabels = en_tokens, yticklabels = es_tokens)
            plt.xticks(rotation=45)

    image = BytesIO()
    fig.tight_layout()
    fig.savefig(image, format='jpeg')
    return base64.b64encode(image.getvalue()).decode('utf-8').replace('\n', '')

def get_code_list():
    code_list = []
    for file in glob.glob(CODE_DIR + "/*.en"):
        code = file.split('/')[1].split('.')[0]
        code_list.append(code)
    return code_list


def shuffle_file(selected_code):
    code_list = get_code_list()
    selected_code = selected_code.lower()
    # print(str(selected_code))
    # print(str(code_list))
    if selected_code in code_list:
        filepath = CODE_DIR + "/" + selected_code + '.en'
        # print(filepath)
        df = pd.read_csv(filepath, header=None)
        data = df.sample(frac=1).values.tolist()
        return data
    else:
        print("No file exists!")
        return False

def vocab_exist(token_list, model_dir=VOCAB_DIR, language="en", check_tag = False):
    # check if this word show up in the vocab file
    # :params: model_dir: directory to the model
    # :params: language: the language of the input token list
    # :params: check_tag: if check whether this word belongs to the nfpa/euro vocab, if not, only check whether this word belongs to share vocab
    if check_tag == True:
        merge_vocab = {"en": pd.read_csv(os.path.join(model_dir, "encoder-merge-NFPA_CS_Train_Vocab.en"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens"), \
                       "es": pd.read_csv(os.path.join(model_dir, "decoder-merge-NFPA_CS_Train_Vocab.es"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens")}
        nfpa_vocab = {"en": pd.read_csv(os.path.join(model_dir, "NFPA_CS_Train_Vocab.en"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens"), \
                      "es": pd.read_csv(os.path.join(model_dir, "NFPA_CS_Train_Vocab.es"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens")}
        euro_vocab = {"en": pd.read_csv(os.path.join(model_dir, "Europarl_Train_Vocab.en"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens"), \
                      "es": pd.read_csv(os.path.join(model_dir, "Europarl_Train_Vocab.es"), delimiter="\t", header=None) \
            .rename(columns = {0: "tokens"}).set_index("tokens")}
        token_tag_dict = {token: (token in merge_vocab[language].index, \
                                  token in nfpa_vocab[language].index, \
                                  token in euro_vocab[language].index) for token in token_list}
        token_tag_dict = {k: "shared" if all([v[1], v[2]]) else ("nfpa" if all([v[1], not v[2]]) else ("euro" if all([not v[1], v[2]]) else None)) for k,v in token_tag_dict.items()}
        token_tag_dict_strip = {k.lstrip("_"): v for (k,v) in token_tag_dict.items()}
        nfpa = [k for k,v in token_tag_dict.items() if v == "nfpa"]
        euro = [k for k,v in token_tag_dict.items() if v == "euro"]
        share = [k for k,v in token_tag_dict.items() if v == "shared"]
        oov = [k for k,v in token_tag_dict.items() if v == None]
        result = {"token_list": token_tag_dict, "striped_token_list": token_tag_dict_strip, "nfpa": nfpa, "euro": euro, "share": share, "oov":oov}
        return result
    else:
        merge_vocab = {"en": pd.read_csv(os.path.join(model_dir, "encoder-merge-NFPA_CS_Train_Vocab.en"), delimiter="\t", header=None) \
            .reset_index().rename(columns = {0: "tokens", "index": "merge_idx"}).set_index("tokens"), \
                       "es": pd.read_csv(os.path.join(model_dir, "decoder-merge-NFPA_CS_Train_Vocab.es"), delimiter="\t", header=None) \
            .reset_index().rename(columns = {0: "tokens", "index": "merge_idx"}).set_index("tokens")}
        token_tag_dict = {token: token in merge_vocab[language].index for token in token_list}
        token_tag_dict_strip = {k.lstrip("_"): v for (k,v) in token_tag_dict.items()}
        iv = [k for k,v in token_tag_dict.items() if v == True]
        oov = [k for k,v in token_tag_dict.items() if v == False]
        result = {"token_list": token_tag_dict, "striped_token_list": token_tag_dict_strip, "iv": iv, "oov": oov}
        return result


parser = argparse.ArgumentParser(description="Translation client example")
parser.add_argument("--model_name", required=True,
                    help="model name")
parser.add_argument("--bpe_model", default="nfpa_models/nfpa_new/1550597313/assets.extra/merged_en_es.bpe",
                    help="path to the BPE model")
parser.add_argument("--host", default="localhost",
                    help="model server host")
parser.add_argument("--port", type=int, default=9000,
                    help="model server port")
parser.add_argument("--timeout", type=float, default=60.0,
                    help="request timeout")
args = parser.parse_args()

channel = grpc.insecure_channel("%s:%d" % (args.host, args.port))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# tokenizer with BPE model
# tokenizer = pyonmttok.Tokenizer("space", joiner_annotate=True, segment_numbers=True, segment_alphabet_change=True, bpe_model_path=args.bpe_model)

# tokenizer for nfpa model
# tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True, segment_numbers=True, segment_alphabet_change=True)

# tokenizer for cliang model
# tokenizer = pyonmttok.Tokenizer("aggressive", spacer_annotate=True)

# tokenizer for multi-language mode
tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)


# Debugging Sample
batch_input = "france passed the budget."
# batch_input = "France."
batch_output = translate(stub, args.model_name, batch_input, tokenizer, timeout=args.timeout)
# model_list = ['nfpa', 'cliang']
# correction(stub, model_list, batch_input, tokenizer, timeout=args.timeout)
# print("")
# with open('test.txt', 'r') as f:
#     data = f.readlines()
# batch_output = translate_file(stub, args.model_name, data, tokenizer, timeout=5.0)
# print("Raw Input")
# print(data)
# print("Output")
# for output_text in batch_output:
#     print(output_text)
