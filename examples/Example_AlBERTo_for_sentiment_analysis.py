import datetime
import sys
import warnings
warnings.filterwarnings("ignore")

#for code working
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import numpy as np

#Prepare and import BERT modules
import subprocess
subprocess.call(["git", "clone", "https://github.com/google-research/bert","bert_repo"])

if not 'bert_repo' in sys.path:
  sys.path += ['bert_repo']

# import python modules defined by BERT
from run_classifier import *
import modeling
import tokenization

#Test transformation
test_data = "Sei un imbecille brutto cattivo"
print(test_data)
sentences = [test_data]
examples_test = []

#Inizialize Text preprocessor

text_processor = TextPreProcessor (
    # terms that will be normalized
    normalize=[ 'url' , 'email', 'user', 'percent', 'money', 'phone', 'time', 'date', 'number'] ,
    # terms that will be annotated
    annotate={"hashtag"} ,
    fix_html=True ,  # fix HTML tokens

    unpack_hashtags=True ,  # perform word segmentation on hashtags

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts = [ emoticons ]
)

examples_test = []
i = 0
for s in sentences:
    s = s.lower()
    s = str(" ".join(text_processor.pre_process_doc(s)))
    s = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]", ' ', s)
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r'(\w)\1{2,}',r'\1\1', s)
    s = re.sub ( r'^\s' , '' , s )
    s = re.sub ( r'\s$' , '' , s )
    print("Processing:---> "+s)
    examples_test.append([0, s])
    i = i+1

examples_test = np.array(examples_test)

f2 = lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[1],
                                                                   text_b = None,
                                                                   label = 0)

test_examples = map(f2,examples_test)
test_examples = list(test_examples)
label_list = [0,1]

#Inizialize the tokenizer
tokenizer = tokenization.FullTokenizer("HASPEEDE_TASK2_r3/models/vocabulary_lower_case_128.txt", do_lower_case=True)
tokenizer.tokenize("Che bella giornata oggi! Stiamo procedendo bene o no? <url>")

#Inizialize AlBERTo
INIT_CHECKPOINT = 'HASPEEDE_TASK2_r3/models/model.ckpt-88'
#SET THE PARAMETERS
PREDICT_BATCH_SIZE = 512
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5
BERT_CONFIG= modeling.BertConfig.from_json_file("HASPEEDE_TASK2_r3/models/bert_config.json")

#Prepare predictions
input_features = convert_examples_to_features(
      test_examples,label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn = input_fn_builder(
    features=input_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

model_fn = model_fn_builder(
  bert_config=BERT_CONFIG,
  num_labels=2,
  init_checkpoint=INIT_CHECKPOINT,
  learning_rate=LEARNING_RATE,
  num_train_steps=0,
  num_warmup_steps=0,
  use_tpu=False,
  use_one_hot_embeddings=True
)

def get_run_config(output_dir):
  return tf.contrib.tpu.RunConfig(
    model_dir="HASPEEDE_TASK2_r3/output")

estimator = tf.contrib.tpu.TPUEstimator(
  use_tpu=False,
  model_fn=model_fn,
  predict_batch_size=PREDICT_BATCH_SIZE,
  config=get_run_config("HASPEEDE_TASK2_r3/output")
)

predictions = estimator.predict(predict_input_fn)

print('***** Finished evaluation at {} *****'.format(datetime.datetime.now()))
for example, prediction in zip(sentences,predictions):
    print('\t prediction:%s \t text_a: %s' % ( np.argmax(prediction['probabilities']),str(example) ) )