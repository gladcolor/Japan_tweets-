# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd
from IPython.display import HTML, display

import pandas as pd
import smart_open
import numpy as np 
from sklearn import preprocessing

import emoji

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.classification import ClassificationModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


pd.set_option('display.max_colwidth', 0)



from sklearn.model_selection import train_test_split



pd.set_option('display.max_colwidth', 0)






def clean_twts(tw):

  # remove urls
  tw = str(tw)
  pattern = 'https{0,1}:\/\/t.co\/[a-zA-Z0-9]+'
  tw = re.sub(pattern, "", tw)

  # remove @
  pattern = '@[a-zA-Z0-9_]+ '
  tw = re.sub(pattern, "", tw)

  # remove emojis
  tw = emoji.demojize(tw)

  return tw

def clean_twt_row(row):
  return clean_twts(row['message'])

# all_sample_file = r'https://raw.githubusercontent.com/gladcolor/tweet_classification/master/samples_8400_demojized.zip'

# samples_df = pd.read_csv(all_sample_file)

def get_positive_sample():
    train_sample_file = r'/content/drive/MyDrive/tweet_classification/d_train.xlsx'
    train_samples_df = pd.read_excel(train_sample_file)

    test_sample_file = r'/content/drive/MyDrive/tweet_classification/d_test.xlsx'
    test_samples_df = pd.read_excel(test_sample_file)
    
    return train_samples_df, test_samples

# test_sample_file = r'/content/drive/MyDrive/tweet_classification/suicide_twts339k.zip'
# test_samples_df = pd.read_csv(test_sample_file, engine='c', encoding='utf16')


# remove emojis
# train_samples_df['message'] = train_samples_df['message'].apply(emoji.demojize)
# test_samples_df['message'] = test_samples_df['message'].apply(emoji.demojize)

def get_train_sample_Japanese():    
    # positive tweet only
    # samples_df1 = pd.read_csv(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/week2_positive1083.csv')
    samples_df1 = pd.read_csv(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\week2_positive1083.csv')
    

    # has positive and negative
    # samples_df2 = pd.read_csv(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/「１」LIZ_2021-12-10T12_49_21.000Z_9492.csv')
    samples_df2 = pd.read_csv(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\「１」LIZ_2021-12-10T12_49_21.000Z_9492.csv')

    # positive only
    # samples_df3 = pd.read_csv(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/positive_5087.csv')
    samples_df3 = pd.read_csv(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\positive_5087.csv')

    # all negative
    # samples_df4 = pd.read_excel(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/TotalNegative.xlsx')
    samples_df4 = pd.read_excel(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\TotalNegative.xlsx')

    # positive
    # samples_df5 = pd.read_excel(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/SuicideRelate_Round2_BasedOn20000.xlsx')
    samples_df5 = pd.read_excel(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\SuicideRelate_Round2_BasedOn20000.xlsx')

    # negative from LIZ's second' file
    # samples_df6 = pd.read_csv(r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/LIZ_negative_2021-12-20T11_42_05.000Z_2021-12-21T02_34_19.000Z_9385.csv')
    samples_df6 = pd.read_csv(r'T:\Shared drives\T5\Japan_Tweets_2021_01_01_2022_01_01\LIZ_negative_2021-12-20T11_42_05.000Z_2021-12-21T02_34_19.000Z_9385.csv')

    samples_df6 = samples_df6[['message']]
       
    # positive in samples_df2
    positive_idx = samples_df2['RISK(YES:1,NO:0)'] == 1
    samples_df2_positive = samples_df2[positive_idx][['id', 'text']]
    samples_df2_positive['message'] = samples_df2_positive['text']
    samples_df2_positive = samples_df2_positive.drop(columns=['text', 'id'])

    # negative in samples_df2
    samples_df2_negative = samples_df2[~positive_idx][['id', 'text']]
    samples_df2_negative['message'] = samples_df2_negative['text']
    samples_df2_negative = samples_df2_negative.drop(columns=['text', 'id'])    
    
    samples_df4['message'] = samples_df4['CONTENT']
    samples_df4 = samples_df4.drop(columns=['CONTENT', 'TWEETID'])

    samples_df5 = samples_df5[['message']]

    positive_df = pd.concat([samples_df1, samples_df2_positive, samples_df3, samples_df5]).sample(frac=1, random_state=42)  # samples_df1, samples_df2_positive, ,
    negative_df = pd.concat([samples_df2_negative, samples_df6]).dropna().sample(frac=1, random_state=42)   # samples_df4, samples_df2_negative, samples_df6
    positive_df = positive_df.sample(frac=1, random_state=42) 
    negative_df = negative_df.sample(frac=1, random_state=42)    


    # remove the "自殺" tweets
    # keywords = ['首吊り', '首を吊る', '首つり', '死ぬ気', '自分を傷つける', 'この世を去る', '死ぬに値する', '自分の人生を終わらせたいという願望', '死にたい', '自傷', '私の命を奪う', '死にたい', '死にたいです', '私の遺書', '私の人生を終わらせる', '決して起きない', '生きる価値がない', '飛び降りる', '永遠に眠る', '電車に飛び込む', '私がいないほうがいい', '生きるのに疲れた', '一人で死ぬ', '永遠に眠る', '私の悲しい人生', 'ストレスを感じる', 'ストレスで参っている', '感情の起伏が激しい', '私自身が嫌い', '精神的に弱い', '練炭', '焼身', '服毒', 'もう死にたい', '自殺サイト楽に死ねる方法', '生きることがつらい', '死にたい 助けて', '安楽死方法', '一番楽に死ねる方法', '簡単に死ねる方法', '消えたい', '確実に死ねる方法', '生きる意味が分からない', 'うつ 死にたい']
    # pattern = '|'.join(keywords)  #"自殺"  "自殺", 
    # mask_ids = negative_df['message'].str.contains(pattern)#.sum()
    # negative_df = negative_df[mask_ids]

    positive_df['message'] = positive_df.apply(clean_twt_row, axis=1)
    negative_df['message'] = negative_df.apply(clean_twt_row, axis=1)

    positive_df['Label'] = 1
    negative_df['Label'] = 0

    # print("Negative count no 自殺:", len(negative_df))

    # select negative sample for test set in a natural distribution.
    test_positive_ratio = 1
    test_ratio = 0.2    
    test_negative_cnt = int(len(negative_df) * test_ratio)   
    test_positive_cnt = int(len(positive_df) * test_ratio)
    print("test_positive_cnt, test_negative_cnt:", test_positive_cnt, test_negative_cnt)

    test_positive_df = positive_df.iloc[:test_positive_cnt]
    test_negative_df = negative_df.iloc[:test_negative_cnt]

    train_positive_df = positive_df.iloc[test_positive_cnt:]
    train_negative_df = negative_df.iloc[test_negative_cnt:]
     
    test_df = pd.concat([test_positive_df, test_negative_df]).sample(frac=1, random_state=42)
    train_df = pd.concat([train_positive_df, train_negative_df]).sample(frac=1, random_state=42)
 
    # print("Negative count int natural_dist_positive_df:", test_positive_cnt)
    # print("Negative count int natural_dist_negative_df no duplicates:", len(natural_dist_positive_df.drop_duplicates(subset='message')))

    print("\nPositive count:", len(positive_df))
    print("Positive count no duplicates:", len(positive_df.drop_duplicates(subset='message')))

    print("\nNegative count:", len(negative_df))
    print("Negative count no duplicates:", len(negative_df.drop_duplicates(subset='message')))

    print("\nTraining set sample count (before over-sampling positive tweets):", len(train_df))
    print("Training set positive count:", len(train_df.query("Label == 1")))
    print("Training set negative count:", len(train_df.query("Label == 0")))
    
    # over sampling
    train_positive_df = train_positive_df.sample(n=int(len(train_negative_df) * 3), replace=True, random_state=42)
    train_df = pd.concat([train_positive_df, train_negative_df]).sample(frac=1, random_state=42)

    print("Train set positive count no duplicates:", len(train_positive_df.drop_duplicates(subset='message')))
    print("Train set negative count no duplicates:", len(train_negative_df.drop_duplicates(subset='message')))


    test_positive_df = test_positive_df.sample(n=int(len(test_negative_df) * 1), replace=True, random_state=42)
    test_df = pd.concat([test_positive_df, test_negative_df]).sample(frac=1, random_state=42)

    print("\nTraining set sample count (after over-sampling positive tweets):", len(train_df))
    print("Training set positive count:", len(train_df.query("Label == 1")))
    print("Training set negative count:", len(train_df.query("Label == 0")))

    print("\nTest set sample count:", len(test_df))
    print("Test set positive count:", len(test_df.query("Label == 1")))
    print("Test set negative count:", len(test_df.query("Label == 0")))
    print("")

    print("Training set contains positive tweets in test set:", test_positive_df['message'].isin(train_positive_df['message'].to_list()).sum())
    print("Training set contains negative tweets in test set:", test_negative_df['message'].isin(train_negative_df['message'].to_list()).sum())
    
    print("Test set positive count no duplicates:", len(test_positive_df.drop_duplicates(subset='message')))
    print("Test set negative count no duplicates:", len(test_negative_df.drop_duplicates(subset='message')))

    return train_df, test_df
# make a test
train_samples_df, test_samples_df  = get_train_sample_Japanese()



from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model_name =  'cl-tohoku/bert-base-japanese-v2'
# model_name = r'cl-tohoku/bert-large-japanese'
# model_name = 'bert-large-uncased'

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# model = SentenceTransformer(model_name)#.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def train_a_model(output_dir, train_df, test_df):
    train_df['Label'] = train_df['Label'].astype(int)
    test_df['Label'] = test_df['Label'].astype(int)

    train_df = train_df.sample(frac=1)
    
    num_labels = 2
    num_train_epochs = 3
    model = ClassificationModel(
    model_type = "bert",
    # model_name = r'bert-base-uncased',  
    model_name = model_name,
    # model_name = r'/content/drive/MyDrive/tweet_classification/outputs',
    num_labels=num_labels, 
    args={"reprocess_input_data": True,   # 对输入数据进行预处理
          "overwrite_output_dir": True,    # 可覆盖输出文件夹
          "save_steps": -1,
          "save_model_every_epoch": False,
          "output_dir": output_dir,
          "train_batch_size": 8*8, # default: 8
          "eval_batch_size": 64*8, # default: 8
          },  
    weight=[1, 1],
    cuda_device=0,
    )

    model.tokenizer = tokenizer

    
    model.train_model(train_df,args = {'fp16':True, 
        "num_train_epochs": num_train_epochs})

    
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
     

    print(f"In the reported positives, about {(100 * result['tp'] / (result['tp'] + result['fp'])):.0f}% are correct, i.e., {result['tp']}/({result['tp']}+{result['fp']}).")
    print(f"Precision: {(result['tp'] / (result['tp'] + result['fp'])):.3f}")
    print("")

    print(f"In the reported positives, contain about {(100 * result['tp'] / (result['fn'] + result['tp'])):.0f}% true positive tweets, i.e., {result['tp']}/({result['fn']}+{result['tp']}).")
    print(f"Recall: {(result['tp'] / (result['tp'] + result['fn'])):.3f}")

    model.save_model(output_dir=output_dir, results=result)

    return result, model_outputs, wrong_predictions


output_dir = r'/content/drive/Shareddrives/T5/Japan_Tweets_2021_01_01_2022_01_01/Trained_model_all'
result, model_outputs, wrong_predictions = train_a_model(output_dir, train_samples_df, test_samples_df)


