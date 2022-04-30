#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download('twitter_samples')


# In[4]:


from nltk.corpus import twitter_samples


# In[5]:


from nltk.corpus import twitter_samples 
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')


# In[6]:


import nltk
nltk.download('punkt')


# In[7]:


from nltk.corpus import twitter_samples 
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json') 
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')


# In[8]:


from nltk.corpus import twitter_samples 
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json') 
text = twitter_samples.strings('tweets.20150430-223406.json') 
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
print(tweet_tokens[0])


# In[9]:


from nltk.corpus import twitter_samples 
positive_tweets = twitter_samples.strings('positive_tweets.json') 
negative_tweets = twitter_samples.strings('negative_tweets.json') 
text = twitter_samples.strings('tweets.20150430-223406.json') 
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
 #print(tweet_tokens[0])


# In[ ]:


Normalizando dados


# In[12]:


... 
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []    
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence 
print(lemmatize_sentence(tweet_tokens[0]))


# In[ ]:


Removendo ruídos dos dados


# In[32]:


...

import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# In[34]:


import nltk


# In[36]:


import nltk
nltk.download('stopwords')


# In[37]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english') 
print(remove_noise(tweet_tokens[0], stop_words))


# In[38]:


...
from nltk.corpus import stopwords
stop_words = stopwords.words('english') 
#print(remove_noise(tweet_tokens[0], stop_words)) 
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []
for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


# In[40]:


...
print(positive_tweet_tokens[500]) 
print(positive_cleaned_tokens_list[500])


# In[ ]:


Determinando a densidade da palavra


# In[41]:


... 
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token 
all_pos_words = get_all_words(positive_cleaned_tokens_list)


# In[42]:


from nltk import FreqDist 
freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))


# In[ ]:


Preparando dados para o modelo


# In[43]:


...
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens) 
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)


# In[ ]:


Divisão do conjunto de dados para treinamento e teste do modelo


# In[44]:


...
import random 
positive_dataset = [(tweet_dict, "Positive")for tweet_dict in positive_tokens_for_model] 
negative_dataset = [(tweet_dict, "Negative")for tweet_dict in negative_tokens_for_model] 
dataset = positive_dataset + negative_dataset 
random.shuffle(dataset) 
train_data = dataset[:7000]
test_data = dataset[7000:]


# In[ ]:


Criando e testando o modelo


# In[45]:


...
from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data) 
print("Accuracy is:", classify.accuracy(classifier, test_data)) 
print(classifier.show_most_informative_features(10))


# In[46]:


...
from nltk.tokenize import word_tokenize 
custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
custom_tokens = remove_noise(word_tokenize(custom_tweet)) 
print(classifier.classify(dict([token, True] for token in custom_tokens)))


# In[49]:


...
custom_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'


# In[50]:


...
custom_tweet = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'


# In[ ]:




