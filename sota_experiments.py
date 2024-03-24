# !pip install vaderSentiment --quiet
# !pip install textstat --quiet
# !pip install nltk --quiet
# !pip install tpot --quiet
# !pip install seaborn --quiet

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
import random
import config
from nltk.tokenize import TweetTokenizer
import mosestokenizer
import numpy as np
from sklearn.model_selection import StratifiedKFold

from random import sample

from tpot import TPOTClassifier

# winner tpot-pipeline 40
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import json

from config import LSHS_DATAFILE, VIDH_DATAFILE, HEVAL_DATAFILE

import os
import shutil

from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torchmetrics
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stopwords=stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()
sentiment_analyzer = VS()

# supporting functions
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""


    #############LINE FIXED: * REPLACED WITH +##################### PREVIOUS::: tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""

    #############LINE FIXED: * REPLACED WITH +##################### PREVIOUS::: tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
    return tweet.split()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet) #Get text only
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


class Features:
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,
        use_idf=True,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=10000,
        min_df=5,
        max_df=0.75
        )
        
        #We can use the TFIDF vectorizer to get a token matrix for the POS tags
        self.pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
        )
        
        self.other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]
        
    def __tfidf__(self, tweets, isTrain=True):
        #Construct tfidf matrix and get relevant scores
        if isTrain:
            return self.vectorizer.fit_transform(tweets).toarray()
        else:
            return self.vectorizer.transform(tweets).toarray()
    
    def __get_pos_tags__(self, tweets):
        #Get POS tags for tweets and save as a string
        tweet_tags = []
        for t in tweets:
            tokens = basic_tokenize(preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)
        return tweet_tags
    
    def __pos_tags__(self, tweets, isTrain=True):
        tweet_tags = self.__get_pos_tags__(tweets)
        
        #Construct POS TF matrix and get vocab dict
        if isTrain:
            return self.pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
        else:
            return self.pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()
    
    def get_features(self, tweets, isTrain=True):
        tfidf = self.__tfidf__(tweets, isTrain=isTrain)
        pos = self.__pos_tags__(tweets, isTrain=isTrain)
        self.feats = get_feature_array(tweets)
        
        #Now join them all up
        # recover ids for mapping
        # ids = np.array(tweets.index.to_list())
        # ids = ids.reshape(ids.shape[0], 1)
        # M = np.concatenate([ids, tfidf,pos,feats],axis=1)
        M = np.concatenate([tfidf, pos, self.feats],axis=1)
        
        X = pd.DataFrame(M)
        
        return X

# cf related functions

def get_offensive_words():
    _df = pd.read_csv(config.en_swear_words_datafile, index_col=0)
    
    s = np.logical_or(_df['Level of offensiveness']=='Strongest words', _df['Level of offensiveness']=='Strong words')
    # display(_df[s]['Word'].to_list())
    wd_list = _df['Word'].to_list()
    
    _df = pd.read_csv(config.en_profanity_datafile, index_col=None)
    s = _df['severity_description'] == 'Severe'
    # wd_list.extend(_df[s]['text'].to_list())
    wd_list.extend(_df['text'].to_list())
    wd_list = set(map(str.lower, wd_list))
    return wd_list

def find_phrases(tokens, phrases):
    tokens = list(map(str.lower, tokens))
    """
    Find phrases in a list of sequential tokens.
    
    Args:
        tokens (list): List of sequential tokens.
        phrases (list): List of phrases to search for.
        
    Returns:
        A list of tuples containing the start and end index of each found phrase.
    """
    found_phrases = []
    
    for i in range(len(tokens)):
        for phrase in phrases:
            if tokens[i:i+len(phrase)] == phrase:
                found_phrases.append((i, i+len(phrase)-1))
    
    return found_phrases

def offensive_lexicon_used(t):
    tk = TweetTokenizer()
    detk = mosestokenizer.MosesDetokenizer('en')
    tk = tk.tokenize(t)
    # print(tk)
    phrase_index = find_phrases(tk, list(map(str.split, offensive_wd_list)))
    return len(phrase_index)


def __exp__(pipeline, _x_train, _x_test, _y_train, _y_test, CF=False):
    f = Features()
    training_features = f.get_features(_x_train, isTrain=True)
    testing_features = f.get_features(_x_test, isTrain=False)
    
    if not CF:
        print('> Train samples', _x_train.shape[0])
    else:
        print('> Train with CF samples', _x_train.shape[0])
    
    try:
        # Fix random state for all the steps in exported pipeline
        set_param_recursive(pipeline.steps, 'random_state', 42)
    except:
        pass

    pipeline.fit(training_features, _y_train)
    results = pipeline.predict(testing_features)

    # report = classification_report(_y_test, results)
    # print(report)
    acc = accuracy_score(_y_test, results)
    f1_marco = f1_score(_y_test, results, average='macro')
    f1_weighted = f1_score(_y_test, results, average='weighted')
    f1_non_avg = f1_score(_y_test, results, average=None)
    r = {'Accuracy': acc,
         'F1-Macro': f1_marco,
         'F1-Weighted': f1_weighted,
         'F1_Class 0': f1_non_avg[0],
         'F1_Class 1': f1_non_avg[1],
         'F1_Class 2': f1_non_avg[2],
        }
    print(r)
    return [r]

def get_tpot_pipeline():
    return make_pipeline(
        StackingEstimator(estimator=LinearSVC(C=15.0, dual=False, loss="squared_hinge", penalty="l2", tol=0.01)),
        DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=12, min_samples_split=3)
    )
    
def get_davidson_pipeline():
    # CHANGE --> solver='liblinear' added for l1, otherwise won't work
    # CHANGE --> max_iter=10000 added for l2, otherwsie ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    return Pipeline([('sel-lg-l1', SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01, solver='liblinear'))),
                     ('lg-l2',LogisticRegression(class_weight='balanced',penalty='l2',C=0.01,  max_iter=10000))])
        
# main experiment
def run_experiment_cross_val_tradtional_ml(cf_tweets_df, tpot_lshs_sota, davidson_pipline_lshs_sota):
    df = pd.read_csv(LSHS_DATAFILE)
    domains = df['Domain'].unique().tolist()
    for d in domains:
        print(d, df[df['Domain'] == d].shape)
        
    df.head()
    
    # Strategy
    CF_LABEL = 0
    random.seed(42)

    print(df['Label'].value_counts())
    problematic_df = df[df['Label']>0]
    problematic_df.shape

    def get_tweets(cf_tweets_df):
        auto_counterfactual_tweets = {}
        tot = problematic_df.shape[0]
        # print(tot)
        for i in range(0, tot):
            idx = problematic_df.iloc[i].name
            if str(i) in cf_tweets_df:
                auto_counterfactual_tweets[idx] = cf_tweets_df[str(i)]
        return auto_counterfactual_tweets

    auto_counterfactual_tweets = get_tweets(cf_tweets_df)
    print(len(auto_counterfactual_tweets))
    
    offensive_wd_list = get_offensive_words()

    def get_counterfactual_tweets(data, labels, cf_label, single_cf_per_tweet=False, cf_size_prop_to_data=1.0, only_tweets_with_offensive_lexicon=True):
        tweets = []
        cnt =0 
        for idx in data.index:
            if idx in auto_counterfactual_tweets:
                if (not only_tweets_with_offensive_lexicon) or offensive_lexicon_used(X[idx]):
                    cnt += 1
                    if not single_cf_per_tweet:
                        tweets.extend(auto_counterfactual_tweets[idx])
                    else:
                        tweets.append(sample(auto_counterfactual_tweets[idx], 1)[0])                     
                        # sample(list1,3)
        print('> Total Tweets used to generate counterfactuals ' + str(cnt))
        print('> Total counterfactuals added ' + str(len(tweets)))
        k = round(cf_size_prop_to_data * len(tweets))
        
        tweets = random.sample(tweets, k=k)
        print('> Counterfactual size ' + str(k) + ' at rate ' + str(cf_size_prop_to_data))
        cf_target = k*[cf_label]
        return pd.concat([data, pd.Series(tweets)], axis=0), pd.concat([labels, pd.Series(cf_target)], axis=0)

    only_tweets_with_offensive_lexicon = False
    
    def run_experiment_counter_factuals(pipeline, X, y, n_splits=2, cf_size_prop_to_data=0.1):
        out_lst = []
        skf = StratifiedKFold(n_splits=n_splits, random_state=None)
        for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
            # print(splt_idx)
            x_train , x_test = X.iloc[train_index], X.iloc[test_index]
            training_target , testing_target = y.iloc[train_index] , y.iloc[test_index]
    
            # org = __exp__(pipeline, x_train, x_test, training_target, testing_target, CF=False)
    
            x_train_with_cf, training_with_cf_target = get_counterfactual_tweets(
                x_train, training_target, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=cf_size_prop_to_data,
                only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)
    
            cf = __exp__(pipeline, x_train_with_cf, x_test, training_with_cf_target, testing_target , CF=True)
            l = [('splt_idx', splt_idx, len(x_test)),  ('train', len(x_train_with_cf)), {'CF': cf}]
            out_lst.append(l)
        return out_lst
    
    # test functioning of cf splits
    
    for d in domains:
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
        skf = StratifiedKFold(n_splits=2, random_state=None)
        for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
            print(splt_idx)
            X_train , X_test = X.iloc[train_index], X.iloc[test_index]
            y_train , y_test = y.iloc[train_index] , y.iloc[train_index]
            _,_ = get_counterfactual_tweets(
                    X_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=.1, only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)

    for d in domains:
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
        skf = StratifiedKFold(n_splits=3, random_state=None)
        for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
            print(splt_idx)
            X_train , X_test = X.iloc[train_index], X.iloc[test_index]
            training_target , testing_target = y.iloc[train_index] , y.iloc[test_index]
            print(X_train.shape, training_target.shape, X_test.shape, testing_target.shape)
            break
            _,_ = get_counterfactual_tweets(
                x_train, training_target, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=cf_size_prop_to_data,
                only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)

    # main exp
    n_splits=5
    cf_size_prop_to_data_lst = list(np.arange(0.1, 1.1, 0.1))

    out_dict = {}
    for d in domains:
        out_dict[d] = {}
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
        
        # tpot_exported_pipeline = get_tpot_pipeline()
        # res_tpot_lst = run_experiment_org(tpot_exported_pipeline, X, y, n_splits=n_splits)
        # out_dict[d]['Org'] = res_tpot_lst
        out_dict[d]['CF'] = {}
        for cf_size_prop_to_data in cf_size_prop_to_data_lst:
            tpot_exported_pipeline = get_tpot_pipeline()
            
            res_tpot_lst = run_experiment_counter_factuals(tpot_exported_pipeline, X, y, n_splits=n_splits, cf_size_prop_to_data=cf_size_prop_to_data)
            out_dict[d]['CF'][cf_size_prop_to_data] = res_tpot_lst
        json.dump(out_dict, open(tpot_lshs_sota, 'w'))

    for d in domains:
        out_dict[d] = {}
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
    
        # davidson_pipeline = get_davidson_pipeline()
        # res_dav_lst = run_experiment_org(davidson_pipeline, X, y, n_splits=n_splits)
        # out_dict[d]['Org'] = res_dav_lst
        out_dict[d]['CF'] = {}
        for cf_size_prop_to_data in cf_size_prop_to_data_lst:
            davidson_pipeline = get_davidson_pipeline()
        
            res_dav_lst = run_experiment_counter_factuals(davidson_pipeline, X, y, n_splits=n_splits, cf_size_prop_to_data=cf_size_prop_to_data)
            out_dict[d]['CF'][cf_size_prop_to_data] = res_dav_lst
        
        json.dump(out_dict, open(davidson_pipline_lshs_sota, 'w'))

    print('done', cf_size_prop_to_data)
    
# bert ft

def run_experiment_cross_val_bert_ft(cf_tweets_df, ft_lshs_sota):
    NUM_LABELS = 3
    df = pd.read_csv(LSHS_DATAFILE)
    domains = df['Domain'].unique().tolist()
    for d in domains:
        print(d, df[df['Domain'] == d].shape)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    MAX_EPOCHS = 5 #5
    BATCH_SIZE = 16*2 #+ int(55 * 0.9*0.5)
    LEARNING_RATE = 1e-5
    # MODEL_LLM = 'distilbert-base-uncased'
    MODEL_LLM = 'bert-base-uncased'
        
    # Setting the seed
    pl.seed_everything(42, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM)
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    class MyDataset(Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
      def __getitem__(self, idx):
        '''
        encoding.items() -> 
          -> input_ids : [1,34, 32, 67,...]
          -> attention_mask : [1,1,1,1,1,....]
        '''
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
      def __len__(self):
        return len((self.labels))

    class LightningModel(pl.LightningModule):
        def __init__(self, model_name_or_path, num_labels, learning_rate=LEARNING_RATE):
            super().__init__()
    
            self.learning_rate = learning_rate
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
    
            # self.val_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.val_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.val_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.val_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            # self.test_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.test_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.test_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.test_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            self.metrics = {'val': [('val_f1_macro_score', self.val_f1_macro_score), ('val_f1_weighted_score', self.val_f1_weighted_score), ('val_acc', self.val_acc)],
                             'test': [('F1-Macro', self.test_f1_macro_score), ('F1-Weighted', self.test_f1_weighted_score),
                                      ('F1_Class 0', self.test_f1_non_avg_score.cpu()[0], 'test_f1_non_avg_score'), 
                                      ('F1_Class 1', self.test_f1_non_avg_score.cpu()[1], 'test_f1_non_avg_score'),
                                      ('F1_Class 2', self.test_f1_non_avg_score.cpu()[2], 'test_f1_non_avg_score'),
                                      ('Accuracy', self.test_acc)
                                     ]
                            }
            
    
        def forward(self, input_ids, attention_mask, labels):
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("train_loss", outputs["loss"])
            return outputs["loss"]  # this is passed to the optimizer for training
    
        def echo_metrics(self, key, predicted_labels, batch_labels):
            for itm in self.metrics[key]:
                if len(itm) == 2:
                    mt_str, met = itm
                    metric_attribute = None
                else:
                    mt_str, met, metric_attribute = itm
                    # print('this', mt_str, met, metric_attribute)
                met(predicted_labels, batch_labels)
                self.log(mt_str, met, prog_bar=True, metric_attribute=metric_attribute)
            
            
        
        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("val_loss", outputs["loss"], prog_bar=True)
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
    
            self.echo_metrics('val', predicted_labels, batch["labels"])
            # print('#n here->',self.val_f1_non_avg_score(predicted_labels, batch["labels"]).cpu()[0])
            
            
        def test_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.echo_metrics('test', predicted_labels, batch["labels"])
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    CF_LABEL = 0

    random.seed(42)
    
    print(df['Label'].value_counts())
    problematic_df = df[df['Label']>0]
    problematic_df.shape

    def get_tweets(cf_tweets_df):
        auto_counterfactual_tweets = {}
        tot = problematic_df.shape[0]
        # print(tot)
        for i in range(0, tot):
            idx = problematic_df.iloc[i].name
            if str(i) in cf_tweets_df:
                auto_counterfactual_tweets[idx] = cf_tweets_df[str(i)]
        return auto_counterfactual_tweets

    auto_counterfactual_tweets = get_tweets(cf_tweets_df)

    offensive_wd_list = get_offensive_words()

    def get_counterfactual_tweets(data, labels, cf_label, single_cf_per_tweet=False, cf_size_prop_to_data=1.0, only_tweets_with_offensive_lexicon=True):
        tweets = []
        cnt =0 
        for idx in data.index:
            if idx in auto_counterfactual_tweets:
                if (not only_tweets_with_offensive_lexicon) or offensive_lexicon_used(X[idx]):
                    cnt += 1
                    if not single_cf_per_tweet:
                        tweets.extend(auto_counterfactual_tweets[idx])
                    else:
                        tweets.append(sample(auto_counterfactual_tweets[idx], 1)[0])                     
                        # sample(list1,3)
        print('> Total Tweets used to generate counterfactuals ' + str(cnt))
        print('> Total counterfactuals added ' + str(len(tweets)))
        k = round(cf_size_prop_to_data * len(tweets))
        
        tweets = random.sample(tweets, k=k)
        print('> Counterfactual size ' + str(k) + ' at rate ' + str(cf_size_prop_to_data))
        cf_target = k*[cf_label]
        return pd.concat([data, pd.Series(tweets)], axis=0), pd.concat([labels, pd.Series(cf_target)], axis=0)

    only_tweets_with_offensive_lexicon = False

    # test
    for d in domains:
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
        skf = StratifiedKFold(n_splits=2, random_state=None)
        for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
            print(splt_idx)
            X_train , X_test = X.iloc[train_index], X.iloc[test_index]
            y_train , y_test = y.iloc[train_index] , y.iloc[train_index]
            _,_ = get_counterfactual_tweets(
                    X_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=.1, only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)

    # run

    def get_splits(_X, _y, n_splits, test_val_split=0.5):
        skf = StratifiedKFold(n_splits=n_splits, random_state=None)
    
        for splt_idx, (train_index , test_index) in enumerate(skf.split(_X, _y)):
            x_train , x_test = _X.iloc[train_index], _X.iloc[test_index]
            y_train , y_test = _y.iloc[train_index] , _y.iloc[test_index]
        
            x_val, x_test, y_val, y_test = \
                        train_test_split(x_test, y_test, test_size=test_val_split) 
    
            train_texts = x_train.values
            train_labels = y_train.values
            
            valid_texts = x_val.values
            valid_labels = y_val.values
            
            test_texts = x_test.values
            test_labels = y_test.values
            yield splt_idx, (x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels)
    
    def __exp__(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, CF=False):
   
        if not CF:
            print('> Train samples', len(train_texts))
        else:
            print('> Train with CF samples', len(train_texts))
        
        
        train_encodings = tokenizer(list(train_texts), truncation = True, padding = True)
        valid_encodings = tokenizer(list(valid_texts), truncation = True, padding = True)
        test_encodings = tokenizer(list(test_texts), truncation = True, padding = True)
    
        #datasets
        train_dataset = MyDataset(train_encodings, train_labels)
        valid_dataset = MyDataset(valid_encodings, valid_labels)
        test_dataset = MyDataset(test_encodings, test_labels)
        
        #dataloaders
        bs = BATCH_SIZE
        train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle = True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = True, num_workers=4)
        print(len(train_loader), len(valid_loader))
    
        # Setting the seed
        pl.seed_everything(42, workers=True)
        lightning_model = LightningModel(MODEL_LLM, NUM_LABELS)
    
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu",
            devices=1,
            # deterministic=True,
            # log_every_n_steps=30,
            enable_checkpointing=True,  
            logger=False
        )
        
        trainer.fit(model=lightning_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        
        r = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
        del lightning_model
        del trainer
        return r
    
    def run_experiment_counter_factuals(_X, _y, n_splits=2, cf_size_prop_to_data=0.1):
        out_lst = []
        for splt_idx, data_item in get_splits(_X, _y, n_splits):
            # print(splt_idx)
            x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = data_item
    
            x_train_with_cf, y_training_with_cf = get_counterfactual_tweets(
                x_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=cf_size_prop_to_data,
                only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)
            train_texts_cf = x_train_with_cf.values
            train_labels_cf = y_training_with_cf.values
    
            cf = __exp__(train_texts_cf, train_labels_cf, valid_texts, valid_labels, test_texts, test_labels, CF=True)
            l = [('splt_idx', splt_idx, len(test_labels)),  ('train', len(y_training_with_cf)), {'CF': cf}]
            out_lst.append(l)
        return out_lst

    n_splits=5
    cf_size_prop_to_data_lst = list(np.arange(0.1, 1.1, 0.1))
    # n_splits=2
    # cf_size_prop_to_data_lst = list(np.arange(0.1, 0.2, 0.1))
    try:
        del X, y
    except:
        pass

    def run_now(ft_lshs_sota):
        complete_result = []
        out_dict = {}
        for d in domains:
            out_dict[d] = {}
            sel_df = df[df['Domain'] == d]
            print(d, sel_df.shape)
            X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
            
            # res_bert_lst = run_experiment_org(X, y, n_splits=n_splits)
            # out_dict[d]['Org'] = res_bert_lst
            out_dict[d]['CF'] = {}
            for cf_size_prop_to_data in cf_size_prop_to_data_lst:
                res_bert_lst = run_experiment_counter_factuals(X, y, n_splits=n_splits, cf_size_prop_to_data=cf_size_prop_to_data)
                out_dict[d]['CF'][cf_size_prop_to_data] = res_bert_lst
            complete_result.append(out_dict)
            json.dump(out_dict, open('out/'+ MODEL_LLM + '-EP_'+ str(MAX_EPOCHS) + ft_lshs_sota, 'w'))
        return complete_result
    
    start = time.time()
    complete_result = run_now(ft_lshs_sota)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

# exp 2 ood
def run_experiment_ood_bert_ft(cf_tweets_df, ft_lshs_heval_ood_sota, ft_lshs_heval_ood_full_sota):
    df = pd.read_csv(LSHS_DATAFILE)
    domains = df['Domain'].unique().tolist()
    for d in domains:
        print(d, df[df['Domain'] == d].shape)
        
    # Convert to binary labels, combine Offensive and Hate as one class.
    NUM_LABELS = 2
    df['Label'] = df['Label'].replace(2, 1)

    # Labels:0 (nonhate) 1 (hate)
    df_test = pd.read_csv(HEVAL_DATAFILE)
    df_test = df_test.rename(columns={'text': 'Tweet', 'HS': 'Label', 'id': 'TweetID'})

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    MAX_EPOCHS = 5 #5
    BATCH_SIZE = 16*2 #+ int(55 * 0.9*0.5)
    LEARNING_RATE = 1e-5
    # MODEL_LLM = 'distilbert-base-uncased'
    MODEL_LLM = 'bert-base-uncased'
    
    # Setting the seed
    pl.seed_everything(42, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM)
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    class MyDataset(Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
      def __getitem__(self, idx):
        '''
        encoding.items() -> 
          -> input_ids : [1,34, 32, 67,...]
          -> attention_mask : [1,1,1,1,1,....]
        '''
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
      def __len__(self):
        return len((self.labels))

    class LightningModel(pl.LightningModule):
        def __init__(self, model_name_or_path, num_labels, learning_rate=LEARNING_RATE):
            super().__init__()
    
            self.learning_rate = learning_rate
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
    
            # self.val_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.val_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.val_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.val_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            # self.test_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.test_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.test_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.test_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            self.metrics = {'val': [('val_f1_macro_score', self.val_f1_macro_score), ('val_f1_weighted_score', self.val_f1_weighted_score), ('val_acc', self.val_acc)],
                             'test': [('F1-Macro', self.test_f1_macro_score), ('F1-Weighted', self.test_f1_weighted_score),
                                      ('F1_Class 0', self.test_f1_non_avg_score.cpu()[0], 'test_f1_non_avg_score'), 
                                      ('F1_Class 1', self.test_f1_non_avg_score.cpu()[1], 'test_f1_non_avg_score'),
                                      ('Accuracy', self.test_acc)
                                     ]
                            }
            
    
        def forward(self, input_ids, attention_mask, labels):
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("train_loss", outputs["loss"])
            return outputs["loss"]  # this is passed to the optimizer for training
    
        def echo_metrics(self, key, predicted_labels, batch_labels):
            for itm in self.metrics[key]:
                if len(itm) == 2:
                    mt_str, met = itm
                    metric_attribute = None
                else:
                    mt_str, met, metric_attribute = itm
                    # print('this', mt_str, met, metric_attribute)
                met(predicted_labels, batch_labels)
                self.log(mt_str, met, prog_bar=True, metric_attribute=metric_attribute)
            
            
        
        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("val_loss", outputs["loss"], prog_bar=True)
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
    
            self.echo_metrics('val', predicted_labels, batch["labels"])
            # print('#n here->',self.val_f1_non_avg_score(predicted_labels, batch["labels"]).cpu()[0])
            
            
        def test_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.echo_metrics('test', predicted_labels, batch["labels"])
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    CF_LABEL = 0
    random.seed(42)

    print(df['Label'].value_counts())
    problematic_df = df[df['Label']>0]
    problematic_df.shape

    def get_tweets(cf_tweets_df):
        auto_counterfactual_tweets = {}
        tot = problematic_df.shape[0]
        # print(tot)
        for i in range(0, tot):
            idx = problematic_df.iloc[i].name
            if str(i) in cf_tweets_df:
                auto_counterfactual_tweets[idx] = cf_tweets_df[str(i)]
        return auto_counterfactual_tweets

    auto_counterfactual_tweets = get_tweets(cf_tweets_df)

    offensive_wd_list = get_offensive_words()

    def get_counterfactual_tweets(data, labels, cf_label, single_cf_per_tweet=False, cf_size_prop_to_data=1.0, only_tweets_with_offensive_lexicon=True):
        tweets = []
        cnt =0 
        for idx in data.index:
            if idx in auto_counterfactual_tweets:
                if (not only_tweets_with_offensive_lexicon) or offensive_lexicon_used(X[idx]):
                    cnt += 1
                    if not single_cf_per_tweet:
                        tweets.extend(auto_counterfactual_tweets[idx])
                    else:
                        tweets.append(sample(auto_counterfactual_tweets[idx], 1)[0])                     
                        # sample(list1,3)
        print('> Total Tweets used to generate counterfactuals ' + str(cnt))
        print('> Total counterfactuals added ' + str(len(tweets)))
        k = round(cf_size_prop_to_data * len(tweets))
        
        tweets = random.sample(tweets, k=k)
        print('> Counterfactual size ' + str(k) + ' at rate ' + str(cf_size_prop_to_data))
        cf_target = k*[cf_label]
        return pd.concat([data, pd.Series(tweets)], axis=0), pd.concat([labels, pd.Series(cf_target)], axis=0)

    only_tweets_with_offensive_lexicon = False

    # test
    for d in domains:
        sel_df = df[df['Domain'] == d]
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
        skf = StratifiedKFold(n_splits=2, random_state=None)
        for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
            print(splt_idx)
            X_train , X_test = X.iloc[train_index], X.iloc[test_index]
            y_train , y_test = y.iloc[train_index] , y.iloc[train_index]
            _,_ = get_counterfactual_tweets(
                    X_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=.1, only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)


    def __exp__(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, CF=False):
        if not CF:
            print('> Train samples', len(train_texts))
        else:
            print('> Train with CF samples', len(train_texts))
        
        train_encodings = tokenizer(list(train_texts), truncation = True, padding = True)
        valid_encodings = tokenizer(list(valid_texts), truncation = True, padding = True)
        test_encodings = tokenizer(list(test_texts), truncation = True, padding = True)
    
        #datasets
        train_dataset = MyDataset(train_encodings, train_labels)
        valid_dataset = MyDataset(valid_encodings, valid_labels)
        test_dataset = MyDataset(test_encodings, test_labels)
        
        #dataloaders
        bs = BATCH_SIZE
        train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle = True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = True, num_workers=4)
        print(len(train_loader), len(valid_loader))
    
        # Setting the seed
        pl.seed_everything(42, workers=True)
        lightning_model = LightningModel(MODEL_LLM, NUM_LABELS)
    
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu",
            devices=1,
            # deterministic=True,
            # log_every_n_steps=30,
            enable_checkpointing=True,  
            logger=False
        )
        
        trainer.fit(model=lightning_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        
        r = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
        del lightning_model
        del trainer
        return r
    
    def get_splits(_X, _y, _X_test, _y_test, train_size=0.8):
        # 80-20 train-val size
        x_train, x_val, y_train, y_val = \
                        train_test_split(_X, _y, train_size=train_size)
    
        train_texts = x_train.values
        train_labels = y_train.values
        
        valid_texts = x_val.values
        valid_labels = y_val.values
        
        test_texts = _X_test.values
        test_labels = _y_test.values
    
        return x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels
    
   
    def run_experiment_counter_factuals(_X, _y, _X_test, _y_test, cf_size_prop_to_data=0.1):
        out_lst = []
        
        x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = get_splits(_X, _y, _X_test, _y_test)
        x_train_with_cf, y_training_with_cf = get_counterfactual_tweets(
            x_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=cf_size_prop_to_data, 
            only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)
        train_texts_cf = x_train_with_cf.values
        train_labels_cf = y_training_with_cf.values
    
        cf = __exp__(train_texts_cf, train_labels_cf, valid_texts, valid_labels, test_texts, test_labels, CF=True)
        l = [('splt_idx', -1, len(test_labels)),  ('train', len(y_training_with_cf)), {'CF': cf}]
        out_lst.append(l)
        return out_lst

    cf_size_prop_to_data_lst = list(np.arange(0.1, 1.1, 0.1))
    try:
        del X, y
    except:
        pass

    def run_now():
        complete_result = []
        out_dict = {}
        for d in domains:
            out_dict[d] = {}
            sel_df = df[df['Domain'] == d]
            print(d, sel_df.shape)
            X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
    
            X_test, y_test = df_test['Tweet'], df_test['Label'].astype(int)
            
            out_dict[d]['CF'] = {}
            for cf_size_prop_to_data in cf_size_prop_to_data_lst:
                res_bert_lst = run_experiment_counter_factuals(X, y, X_test, y_test, cf_size_prop_to_data=cf_size_prop_to_data)
                out_dict[d]['CF'][cf_size_prop_to_data] = res_bert_lst
            complete_result.append(out_dict)
            json.dump(out_dict, open('out/'+ MODEL_LLM + '-EP_'+ str(MAX_EPOCHS) + ft_lshs_heval_ood_sota, 'w'))
        return complete_result
    
    start = time.time()
    complete_result = run_now()
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    def run_now_full():
        complete_result = []
        out_dict = {}
        d = 'Complete'
        out_dict[d] = {}
        sel_df = df
        print(d, sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
    
        X_test, y_test = df_test['Tweet'], df_test['Label'].astype(int)
        
        out_dict[d]['CF'] = {}
        for cf_size_prop_to_data in cf_size_prop_to_data_lst:
            res_bert_lst = run_experiment_counter_factuals(X, y, X_test, y_test, cf_size_prop_to_data=cf_size_prop_to_data)
            out_dict[d]['CF'][cf_size_prop_to_data] = res_bert_lst
        complete_result.append(out_dict)
        json.dump(out_dict, open('out/'+ MODEL_LLM + '-EP_'+ str(MAX_EPOCHS) + ft_lshs_heval_ood_full_sota, 'w'))
        return complete_result
    
    
    start = time.time()
    complete_result = run_now_full()
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

# exp 3 ood with manual
def run_experiment_ood_with_man_bert_ft(cf_tweets_df, ft_vidh_heval_ood_sota):
    df = pd.read_csv(VIDH_DATAFILE)
    print(df.shape)
    
    # Binary classification in training and testing.
    NUM_LABELS = 2
    df = df.rename(columns={'text': 'Tweet', 'id': 'TweetID'})

    # Labels:0 (nonhate) 1 (hate)
    df_test = pd.read_csv(HEVAL_DATAFILE)
    df_test = df_test.rename(columns={'text': 'Tweet', 'HS': 'Label', 'id': 'TweetID'})

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # https://wandb.ai/jack-morris/david-vs-goliath/reports/Does-Model-Size-Matter-A-Comparison-of-BERT-and-DistilBERT--VmlldzoxMDUxNzU
    MAX_EPOCHS = 5 #5
    BATCH_SIZE = 12 * 1 #+ int(55 * 0.9*0.5)
    LEARNING_RATE = 1e-5
    # MODEL_LLM = 'distilbert-base-uncased'
    MODEL_LLM = 'bert-base-uncased'
    
    # Setting the seed
    pl.seed_everything(42, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LLM)
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    class MyDataset(Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
      def __getitem__(self, idx):
        '''
        encoding.items() -> 
          -> input_ids : [1,34, 32, 67,...]
          -> attention_mask : [1,1,1,1,1,....]
        '''
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
      def __len__(self):
        return len((self.labels))

    class LightningModel(pl.LightningModule):
        def __init__(self, model_name_or_path, num_labels, learning_rate=LEARNING_RATE):
            super().__init__()
    
            self.learning_rate = learning_rate
            self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
    
            # self.val_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.val_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.val_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.val_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            # self.test_conf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=NUM_LABELS)
            self.test_f1_macro_score = torchmetrics.classification.MulticlassF1Score(average="macro", num_classes=NUM_LABELS)
            self.test_f1_weighted_score = torchmetrics.classification.MulticlassF1Score(average="weighted", num_classes=NUM_LABELS)
            self.test_f1_non_avg_score = torchmetrics.classification.MulticlassF1Score(average="none", num_classes=NUM_LABELS)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_LABELS)
            
            self.metrics = {'val': [('val_f1_macro_score', self.val_f1_macro_score), ('val_f1_weighted_score', self.val_f1_weighted_score), ('val_acc', self.val_acc)],
                             'test': [('F1-Macro', self.test_f1_macro_score), ('F1-Weighted', self.test_f1_weighted_score),
                                      ('F1_Class 0', self.test_f1_non_avg_score.cpu()[0], 'test_f1_non_avg_score'), 
                                      ('F1_Class 1', self.test_f1_non_avg_score.cpu()[1], 'test_f1_non_avg_score'),
                                      ('Accuracy', self.test_acc)
                                     ]
                            }
            
    
        def forward(self, input_ids, attention_mask, labels):
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("train_loss", outputs["loss"])
            return outputs["loss"]  # this is passed to the optimizer for training
    
        def echo_metrics(self, key, predicted_labels, batch_labels):
            for itm in self.metrics[key]:
                if len(itm) == 2:
                    mt_str, met = itm
                    metric_attribute = None
                else:
                    mt_str, met, metric_attribute = itm
                    # print('this', mt_str, met, metric_attribute)
                met(predicted_labels, batch_labels)
                self.log(mt_str, met, prog_bar=True, metric_attribute=metric_attribute)
            
            
        
        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            self.log("val_loss", outputs["loss"], prog_bar=True)
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
    
            self.echo_metrics('val', predicted_labels, batch["labels"])
            # print('#n here->',self.val_f1_non_avg_score(predicted_labels, batch["labels"]).cpu()[0])
            
            
        def test_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])        
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.echo_metrics('test', predicted_labels, batch["labels"])
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    CF_LABEL = 0
    random.seed(42)

    print(df['Label'].value_counts())
    problematic_df = df[df['Label']>0]
    problematic_df.shape

    def get_tweets(cf_tweets_df):
        auto_counterfactual_tweets = {}
        tot = problematic_df.shape[0]
        # print(tot)
        for i in range(0, tot):
            idx = problematic_df.iloc[i].name
            if str(i) in cf_tweets_df:
                auto_counterfactual_tweets[idx] = cf_tweets_df[str(i)]
        return auto_counterfactual_tweets

    auto_counterfactual_tweets = get_tweets(cf_tweets_df)

    offensive_wd_list = get_offensive_words()

    def get_counterfactual_tweets(data, labels, cf_label, single_cf_per_tweet=False, cf_size_prop_to_data=1.0, only_tweets_with_offensive_lexicon=True):
        tweets = []
        cnt =0 
        for idx in data.index:
            if idx in auto_counterfactual_tweets:
                if (not only_tweets_with_offensive_lexicon) or offensive_lexicon_used(X[idx]):
                    cnt += 1
                    if not single_cf_per_tweet:
                        tweets.extend(auto_counterfactual_tweets[idx])
                    else:
                        tweets.append(sample(auto_counterfactual_tweets[idx], 1)[0])                     
                        # sample(list1,3)
        print('> Total Tweets used to generate counterfactuals ' + str(cnt))
        print('> Total counterfactuals added ' + str(len(tweets)))
        k = round(cf_size_prop_to_data * len(tweets))
        
        tweets = random.sample(tweets, k=k)
        print('> Counterfactual size ' + str(k) + ' at rate ' + str(cf_size_prop_to_data))
        cf_target = k*[cf_label]
        return pd.concat([data, pd.Series(tweets)], axis=0), pd.concat([labels, pd.Series(cf_target)], axis=0)

    only_tweets_with_offensive_lexicon = False

    # test
    sel_df = df
    print('full dataset', sel_df.shape)
    X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
    skf = StratifiedKFold(n_splits=2, random_state=None)
    for splt_idx, (train_index , test_index) in enumerate(skf.split(X, y)):
        print(splt_idx)
        X_train , X_test = X.iloc[train_index], X.iloc[test_index]
        y_train , y_test = y.iloc[train_index] , y.iloc[train_index]
        _,_ = get_counterfactual_tweets(
                X_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=.1, only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)

    def __exp__(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels, CF=False):
        if not CF:
            print('> Train samples', len(train_texts))
        else:
            print('> Train with CF samples', len(train_texts))
        
        train_encodings = tokenizer(list(train_texts), truncation = True, padding = True)
        valid_encodings = tokenizer(list(valid_texts), truncation = True, padding = True)
        test_encodings = tokenizer(list(test_texts), truncation = True, padding = True)
        
        #datasets
        train_dataset = MyDataset(train_encodings, train_labels)
        valid_dataset = MyDataset(valid_encodings, valid_labels)
        test_dataset = MyDataset(test_encodings, test_labels)
        
        #dataloaders
        bs = BATCH_SIZE
        train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size = bs, shuffle = True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = True, num_workers=4)
        print(len(train_loader), len(valid_loader))
        
        # Setting the seed
        pl.seed_everything(42, workers=True)
        lightning_model = LightningModel(MODEL_LLM, NUM_LABELS)
        chk_pnt_folder = 'checkpoints'
        checkpoint_callback = ModelCheckpoint(dirpath=chk_pnt_folder, save_top_k=1)
        
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu",
            devices=1,
            # deterministic=True,
            # log_every_n_steps=30,
            # enable_checkpointing=True, 
            callbacks=[checkpoint_callback],
            logger=False
        )
        
        trainer.fit(model=lightning_model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        
        r = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")
        del lightning_model
        del trainer
        shutil.rmtree(chk_pnt_folder)
        return r

    def get_splits(_X, _y, _X_test, _y_test, train_size=0.8):
        # 80-20 train-val size
        x_train, x_val, y_train, y_val = \
                        train_test_split(_X, _y, train_size=train_size)
    
        train_texts = x_train.values
        train_labels = y_train.values
        
        valid_texts = x_val.values
        valid_labels = y_val.values
        
        test_texts = _X_test.values
        test_labels = _y_test.values
    
        return x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels
    
    
    def run_experiment_counter_factuals(_X, _y, _X_test, _y_test, cf_size_prop_to_data=0.1):
        out_lst = []
        
        x_train, y_train, train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = get_splits(_X, _y, _X_test, _y_test)
        x_train_with_cf, y_training_with_cf = get_counterfactual_tweets(
            x_train, y_train, cf_label=CF_LABEL, single_cf_per_tweet=True, cf_size_prop_to_data=cf_size_prop_to_data, 
            only_tweets_with_offensive_lexicon=only_tweets_with_offensive_lexicon)
        train_texts_cf = x_train_with_cf.values
        train_labels_cf = y_training_with_cf.values
    
        cf = __exp__(train_texts_cf, train_labels_cf, valid_texts, valid_labels, test_texts, test_labels, CF=True)
        l = [('splt_idx', -1, len(test_labels)),  ('train', len(y_training_with_cf)), {'CF': cf}]
        out_lst.append(l)
        return out_lst

    cf_size_prop_to_data_lst = list(np.arange(0.1, 1.1, 0.1))
    try:
        del X, y
    except:
        pass

    def run_now():
        complete_result = []
        out_dict = {}
        
        sel_df = df
        print('full dataset', sel_df.shape)
        X, y = sel_df['Tweet'], sel_df['Label'].astype(int)
    
        X_test, y_test = df_test['Tweet'], df_test['Label'].astype(int)
        out_dict['CF'] = {}
        for cf_size_prop_to_data in cf_size_prop_to_data_lst:
            res_bert_lst = run_experiment_counter_factuals(X, y, X_test, y_test, cf_size_prop_to_data=cf_size_prop_to_data)
            out_dict['CF'][cf_size_prop_to_data] = res_bert_lst
        complete_result.append(out_dict)
        json.dump(out_dict, open('out/'+ MODEL_LLM + '-EP_'+ str(MAX_EPOCHS) + ft_vidh_heval_ood_sota, 'w'))
        
        return complete_result
    
    start = time.time()
    complete_result = run_now()
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")
    