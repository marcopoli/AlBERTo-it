# AlBERTo the first italian BERT model for Twitter languange understanding
<p align ="justify" style="text-align: justify;">Recent scientific studies on natural language processing (NLP) report the outstanding effectiveness observed in the use of context-dependent and task-free language understanding models such as ELMo, GPT, and BERT. Specifically, they have proved to achieve state of the art performance in numerous complex NLP tasks such as question answering and sentiment analysis in the English language. Following the great popularity and effectiveness that these models are gaining in the scientific community, we trained a BERT language understanding model for the Italian language (<b>AlBERTo</b>). In particular, AlBERTo is focused on the language used in social networks, specifically on Twitter. To demonstrate its robustness, we evaluated AlBERTo on the EVALITA 2016 task SENTIPOLC (SENTIment POLarity Classification) obtaining state of the art results in subjectivity, polarity and irony detection on Italian tweets. The pre-trained  AlBERTo model will be publicly distributed through the GitHub platform at the following web address: https://github.com/marcopoli/AlBERTo-it in order to facilitate future research.</p>
<p align ="center">
<img src="img/AlBERTo.png" width="250"/>
</p>

<h2>Pre-trained model download</h2>

The pre-trained <i>lower cased</i> model on a vocabulary of 128k terms from 200M of tweets can be downloaded here:

*   **[`AlBERTo-Base, Italian Twitter lower cased`](...)**:
    Italian language of social media, 12-layer, 768-hidden, 12-heads, 110M parameters:
    <br><a href="https://storage.googleapis.com/albert_files/alberto_tweets_uncased_L-12_H-768_A-12.zip">alberto_tweets_uncased_L-12_H-768_A-12.zip</a>

<h2>Example of usage</h2>
In order to use the model and run the example "as it is" you need to store AlBERTo in your GCS bucket (https://cloud.google.com/products/storage/). The example is writter to be run on the Google Colab Platform.
<br><br>

*   **[`Fine-Tuning and Classification Task`](AlBERTo_End_to_End_(Fine_tuning_+_Predicting)_with_Cloud_TPU_Sentence_Classification_Tasks.ipynb)**:
    Jupyther notebook used for fine-tuning AlBERTo on the SENTIPOLC 2016 NLP tasks (http://www.di.unito.it/~tutreeb/sentipolc-evalita16/)

<h2>Credits</h2>
<b>Authors</b>:<br> Marco Polignano, Pierpaolo Basile, Marco de Gemmis, Giovanni Semeraro, Valerio Basile

<br><br><b>Cite us:</b>
TBD
