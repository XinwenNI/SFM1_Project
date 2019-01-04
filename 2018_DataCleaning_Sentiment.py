#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:09:35 2018

@author: xinwenni
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import csv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from wordcloud import WordCloud

root_path= os.getcwd()

nasdaq_news      = pd.read_csv('nasdaq_news_2018.csv', sep=',')

def Words(filename = None, header = False, column = 0):
    sent_words = []
    if header == False:  
        try:
            with open(filename, errors = 'ignore') as f:
                        lm = csv.reader(f, delimiter=',')
                        for row in lm:
                            if column != 0:
                                if row[column] != '0':
                                    sent_words.append(row[0])
                            else:
                                sent_words.append(row[0])
        except:
            print('Error: Maybe wrong filename')
    else:
        try:
            with open(filename, errors = 'ignore') as f:
                        lm = csv.reader(f, delimiter=',')
                        for row in lm:
                            if column != 0:
                                if row[column] != '0':
                                    sent_words.append(row[0])
                            else:
                                sent_words.append(row[0])
            sent_words = sent_words[1:]
        except:
            print('Error: Maybe wrong filename')
        
    return(sent_words)

def plot_figures(figures, nrows = 1, ncols=1, name = 'Wordcloud.png'):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.figure(figsize=(10,10))
    fig.savefig(name, dpi = 720,transparent=True)
    
stopwords = Words('en_stopwords.csv')
    # Load the LM-Wordlists in English and German
neg = Words(filename = 'LM_Dict.csv',header = True, column = 7)
pos = Words(filename = 'LM_Dict.csv',header = True, column = 8)


doc = {}
df_raw=nasdaq_news
for i in range (len(df_raw)):
    if "Apple" in df_raw.ix[i,'article_title']:
        tmp_stamp=df_raw.ix[i,'article_time']
        tmp_txt=df_raw.ix[i,'article_content']
        tmp_date=datetime.strptime(str(tmp_stamp[0:10]), '%Y-%m-%d').date()
        tmp_time = tmp_stamp[11:19]
        if tmp_date in doc:
                doc[tmp_date]['Article'].append( (tmp_time,tmp_txt) ) 
        else:
                doc[tmp_date] = {'Article' : [(tmp_time,tmp_txt)]}
        i=i+1
    elif "AAPL" in df_raw.ix[i,'article_title']:
        tmp_stamp=df_raw.ix[i,'article_time']
        tmp_txt=df_raw.ix[i,'article_content']
        tmp_date=datetime.strptime(str(tmp_stamp[0:10]), '%Y-%m-%d').date()
        tmp_time = tmp_stamp[11:19]
        if tmp_date in doc:
                doc[tmp_date]['Article'].append( (tmp_time,tmp_txt) ) 
        else:
                doc[tmp_date] = {'Article' : [(tmp_time,tmp_txt)]}
        i=i+1
    else:
        i=i+1
    

    # create empty DataFrame and fill with nested dict values and keys
ls = []

for key in doc.keys():
    if 'Article' in doc[key]:
        for l in range(len(doc[key]['Article'])):
                    ls.append([key,
                               doc[key]['Article'][l][0],
                               doc[key]['Article'][l][1]])


df         = pd.DataFrame.from_records(ls)
df.columns = ['Date', 'Time', 'Text']
    
df = pd.concat([df, pd.DataFrame(columns = ['PosScore']),
                      pd.DataFrame(columns = ['NegScore']),
                      pd.DataFrame(columns = ['TotalWordCount']),
                      pd.DataFrame(columns = ['SentScore'])])

bow_neg = []
bow_pos = []
    


for i,article in enumerate(df.Text):
    if str(article) != 'nan':
            # Get rid of punctuation
       tmp_word_list    = ' '.join(' '.join(article.split(',')).split('.')).split()
            # Get rid of special characters
       df.Text[i]       = ' '.join([ word for word in tmp_word_list if word.isalnum() ])   
            # Get rid of stopwords
       df.Text[i]       = ' '.join([w for w in df.Text[i].split() if not w.upper() in stopwords])
            # Calculate positive & negative words per article and assign them to DataFrame
       p,n = 0,0
       for word in article.split(' '):
              if word.upper() in pos:
                    p += 1
                    bow_pos.append(word)
              if word.upper() in neg:
                    n += 1
                    bow_neg.append(word)
       df.PosScore[i]       = p
       df.NegScore[i]       = n
       df.TotalWordCount[i] = len(article.split(' '))
       if p == 0 and n == 0:
            df.SentScore[i]  = 0
       else:
            df.SentScore[i]  = (p-n)/(p+n)
df = df.sort_values(['Date','Time']) 

#wordcloud for pos and neg words    
 # Define Mask for stopwords
mask = np.array(Image.open('knowledge_graph_logo.png'))
   
 # Create WordCloud for pos and neg words
wc_neg = WordCloud(background_color="rgba(0, 0, 0, 0)", mode="RGBA", max_words=2000, mask=mask)
fig_neg = wc_neg.generate(' '.join(bow_neg))
wc_pos = WordCloud(background_color="rgba(0, 0, 0, 0)", mode="RGBA", max_words=2000, mask=mask)
fig_pos = wc_pos.generate(' '.join(bow_pos))
wc_com = WordCloud(background_color="rgba(0, 0, 0, 0)", mode="RGBA", max_words=2000, mask=mask)
fig_com = wc_com.generate(' '.join(bow_pos + bow_neg))   
    
    
fig = {'Negative WC': fig_neg,
       'Positive WC': fig_pos,
       'Combined WC': fig_com}
    
plot_figures(fig, 1, 3) 

 
# sentiment analysis for each date
#df_agg            = pd.DataFrame(columns = ['Date','SentScore','Return','Volume', 'DayRG', 'OnRG', 'NoW', 'PrevP'])

df_agg =pd.DataFrame(df.Date.unique())
df_agg.columns = ['Date']

df_agg = pd.concat([df_agg, pd.DataFrame(columns = ['SentScore']),
                 pd.DataFrame(columns = ['Returen']),
                 pd.DataFrame(columns = ['Frequency']),
                 pd.DataFrame(columns = ['Dispersion']),
                 pd.DataFrame(columns = ['Volume'])])


#i=0
#j=0
df = df.sort_values(['Date','Time']) 
df_agg = df_agg.sort_values(['Date']) 
# when the per day article is more than the select value, then calculate the dispersion.
select_value=10
for i in range(len(df_agg)):
    temp_score=0
    temp_num=0
    for j in range(len(df)):
        if df_agg.Date[i]==df.Date[j]:
            temp_score=temp_score+df.SentScore[j]
            temp_num=temp_num+1
    df_agg.SentScore[i]=temp_score/temp_num
    df_agg.Frequency[i]=temp_num
    if temp_num>=select_value:
        df_agg.Dispersion[i]=np.std(df.SentScore[df.Date==df_agg.Date[i]])

#plot the sensitivity (averaged per day)
plt.plot(df_agg.Date,df_agg.SentScore, 'o-',linewidth=1,markersize=2)

plt.title('Sentiment')
plt.xlabel('Time')
plt.ylabel('Sentiment')
plt.gcf().autofmt_xdate()
plt.savefig('Sentiment_Apple.png',dpi = 720,transparent=True)

plt.plot(df_agg.Date,df_agg.Dispersion, 'o-',linewidth=0, markersize=2)

#plot the dispursion and the quantile 
disp_mean = df_agg.Dispersion.mean()
disp_q10 = df_agg.Dispersion.quantile(q=0.1)
disp_q90 = df_agg.Dispersion.quantile(q=0.9)

plt.axhline(y=disp_mean, xmin=0, xmax=1, linewidth=1, color = 'k')
plt.axhline(y=disp_q10, xmin=0, xmax=1, linewidth=1, color = 'r')
plt.axhline(y=disp_q90, xmin=0, xmax=1, linewidth=1, color = 'r')

from datetime import datetime
def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

# create a new data frame to list all dates and store final results 
Datelist=datelist('20180101','20190101')
Datelist=pd.DataFrame(Datelist)
Datelist.columns = ['Date']
for i in range(len(Datelist)):
    Datelist.ix[i,'Date']= datetime.strptime(Datelist.ix[i,'Date'], '%Y-%m-%d').date()


fdf=pd.merge(df_agg,Datelist,how='outer', on='Date')
fdf=fdf.sort_values(['Date'])

# find out the subset of the final dateframe (for matching the finacial data length)
min_date=datetime.strptime('2018-06-01','%Y-%m-%d').date()
sub_fdf=fdf.ix[fdf.Date>=min_date]





