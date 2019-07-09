# AlBERTo the first italian BERT model for Twitter languange understanding
<p align ="justify" style="text-align: justify;">Recent scientific studies on natural language processing (NLP) report the outstanding effectiveness observed in the use of context-dependent and task-free language understanding models such as ELMo, GPT, and BERT. Specifically, they have proved to achieve state of the art performance in numerous complex NLP tasks such as question answering and sentiment analysis in the English language. Following the great popularity and effectiveness that these models are gaining in the scientific community, we trained a BERT language understanding model for the Italian language (<b>AlBERTo</b>). In particular, AlBERTo is focused on the language used in social networks, specifically on Twitter. To demonstrate its robustness, we evaluated AlBERTo on the EVALITA 2016 task SENTIPOLC (SENTIment POLarity Classification) obtaining state of the art results in subjectivity, polarity and irony detection on Italian tweets. The pre-trained  AlBERTo model will be publicly distributed through the GitHub platform at the following web address: https://github.com/marcopoli/AlBERTo-it in order to facilitate future research.</p>
<p align ="center">
<img src="img/AlBERTo.png" width="250"/>
</p>

<h2>Pre-trained model download</h2>

The pre-trained <i>lower cased</i> model on a vocabulary of 128k terms from 200M of tweets can be downloaded here:

*   **[`AlBERTo-Base, Italian Twitter lower cased`](...)**:
    Italian language of social media, 12-layer, 768-hidden, 12-heads, 110M parameters

<h2>Example of usage</h2>
In order to use the model and run the example "as it is" you need to store AlBERTo in your GCS space.
