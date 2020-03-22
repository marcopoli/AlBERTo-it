# 10/09/2019 - Model released

# AlBERTo the first italian BERT model for Twitter languange understanding
<p align ="justify" style="text-align: justify;">Recent scientific studies on natural language processing (NLP) report the outstanding effectiveness observed in the use of context-dependent and task-free language understanding models such as ELMo, GPT, and BERT. Specifically, they have proved to achieve state of the art performance in numerous complex NLP tasks such as question answering and sentiment analysis in the English language. Following the great popularity and effectiveness that these models are gaining in the scientific community, we trained a BERT language understanding model for the Italian language (<b>AlBERTo</b>). In particular, AlBERTo is focused on the language used in social networks, specifically on Twitter. To demonstrate its robustness, we evaluated AlBERTo on the EVALITA 2016 task SENTIPOLC (SENTIment POLarity Classification) obtaining state of the art results in subjectivity, polarity and irony detection on Italian tweets. The pre-trained  AlBERTo model will be publicly distributed through the GitHub platform at the following web address: https://github.com/marcopoli/AlBERTo-it in order to facilitate future research.</p>
Full paper:
http://ceur-ws.org/Vol-2481/paper57.pdf
<p align ="center">
<img src="img/AlBERTo.png" width="250"/>
</p>

<h2>Pre-trained model download</h2>

The pre-trained <i>lower cased</i> model on a vocabulary of 128k terms from 200M of tweets can be downloaded here:

*   **[`AlBERTo-Base, Italian Twitter lower cased`](https://drive.google.com/open?id=15pa20vLqmZDXERbcO73M5GsN0Zo8ittF)**:
    Italian language of social media, 12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`AlBERTo-Base, Italian Twitter lower cased` - Pytorch and Tensorflow Models](https://drive.google.com/open?id=1x1pRE7LZilIcPSWgoNpGyci9NwPqZYiL)**:
    Italian language of social media, 12-layer, 768-hidden, 12-heads, 110M parameters, compatible with Transformers, thanks to Angelo Basile, Junior Research Scientist at Symanto - Profiling AI
    

<h2>Example of usage</h2>
In order to use the model and run the example "as it is" you need to store AlBERTo in your GCS bucket (https://cloud.google.com/products/storage/). The example is writter to be run on the Google Colab Platform.
<br><br>

*   **[`Fine-Tuning and Classification Task`](AlBERTo_End_to_End_(Fine_tuning_+_Predicting)_with_Cloud_TPU_Sentence_Classification_Tasks.ipynb)**:
    Jupyther notebook used for fine-tuning AlBERTo on the SENTIPOLC 2016 NLP tasks (http://www.di.unito.it/~tutreeb/sentipolc-evalita16/)
*   **[`Sentipolc 2016`](https://drive.google.com/open?id=1fODwLwtyKeayk290Tn7s-0-w63Hb-EYL)**:
    Tensorflow fine-tuned models 


<h3>Huggingface.co Transformers <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="20"></h3>

```python
from tokenizer import *
from transformers import AutoTokenizer, AutoModel

a = AlBERTo_Preprocessing(do_lower_case=True)
s: str = "#IlGOverno presenta le linee guida sulla scuola #labuonascuola - http://t.co/SYS1T9QmQN"
b = a.preprocess(s)

tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
tokens = tok.tokenize(b)
print(tokens)

model = AutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
```

<h2>Credits</h2>
<b>Authors</b>:<br> Marco Polignano, Pierpaolo Basile, Marco de Gemmis, Giovanni Semeraro, Valerio Basile

Thanks to Angelo Basile, Junior Research Scientist at Symanto - Profiling AI for the huggingface models.
<br><br><b>Cite us:</b>
```
@InProceedings{PolignanoEtAlCLIC2019,
  author    = {Marco Polignano and Pierpaolo Basile and Marco de Gemmis and Giovanni Semeraro and Valerio Basile},
  title     = {{AlBERTo: Italian BERT Language Understanding Model for NLP Challenging Tasks Based on Tweets}},
  booktitle = {Proceedings of the Sixth Italian Conference on Computational Linguistics (CLiC-it 2019)},
  year      = {2019},
  publisher = {CEUR},
  journal={CEUR Workshop Proceedings},
  volume={2481},
  url={https://www.scopus.com/inward/record.uri?eid=2-s2.0-85074851349&partnerID=40&md5=7abed946e06f76b3825ae5e294ffac14},
  document_type={Conference Paper},
  source={Scopus}
}
```
