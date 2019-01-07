
"""
Created on Sat Jan  5 11:02:53 2019

@author: Xinwenni Ni, Micheal Althof 

The code is for the project of SFM1 WS1819.
The analysis involved mainly two parts: I Sentiment analysis and II Realised Volatility analysis 

"""


import os
from datetime import datetime
import pandas as pd
import csv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from wordcloud import WordCloud

root_path= os.getcwd()

# Predefined functions

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
 

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l  

##############################################
# part I: Sentiment analysis
##############################################

# Predefine some variables 
# when the per day article is more than the select value, then calculate the dispersion.
select_value=10
    
# import stopwords and LM dictionary 
stopwords = Words('en_stopwords.csv')
    # Load the LM-Wordlists in English and German
neg = Words(filename = 'LM_Dict.csv',header = True, column = 7)
pos = Words(filename = 'LM_Dict.csv',header = True, column = 8)

# Import Nasdaq News data
nasdaq_news      = pd.read_csv('nasdaq_news_2018.csv', sep=',')

# Select the subset news related to Apple
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
    

# Clean the text data
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

# wordcloud for pos and neg words    
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
df_agg =pd.DataFrame(df.Date.unique())
df_agg.columns = ['Date']

df_agg = pd.concat([df_agg, pd.DataFrame(columns = ['SentScore']),
                 pd.DataFrame(columns = ['Frequency']),
                 pd.DataFrame(columns = ['Dispersion'])])


df = df.sort_values(['Date','Time']) 
df_agg = df_agg.sort_values(['Date']) 

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



##############################################
# part II: Realised Volatility analysis
##############################################
#function to convert time to same standard across dataframes
def convert_time(row):
   d = datetime.strptime(row, '%d/%m/%Y %H:%M').date()
   d.strftime('%Y-%m-%d')
   return(d)

#NDX = Nasdaq100 5min intervall intraday data, add avg between open and close
NDX          = pd.read_csv('NDX Index.csv', sep=',')

NDX['avg']   =NDX[['Open','Close']].mean(axis=1)
NDX['log_r'] =np.log(NDX['avg']).diff() * np.log(NDX['avg']).diff()
NDX['Date'] = NDX.Dates.apply(convert_time)
NDX = NDX.fillna(NDX.mean())

#add column date without time stamp
Date = NDX['Date']
NDX.drop(labels=['Date'], axis=1,inplace = True)
NDX.insert(0, 'Date', Date) 

#NDX = Nasdaq100 5min intervall intraday data, add avg between open and close
AAPL = pd.read_csv('AAPL US Equity.csv', sep=',')
AAPL['avg']=AAPL[['Open','Close']].mean(axis=1)
AAPL['log_r'] =np.log(AAPL['avg']).diff() * np.log(AAPL['avg']).diff()
AAPL['Date'] = AAPL.Dates.apply(convert_time)
AAPL = AAPL.fillna(AAPL.mean())

#add column date without time stamp
Date = AAPL['Date']
AAPL.drop(labels=['Date'], axis=1,inplace = True)
AAPL.insert(0, 'Date', Date)

#create new data frame with unique dates, sum up intraday squared returns per day
NDX_agg = pd.DataFrame(NDX.Date.unique())
NDX_agg.columns = ['Date']

NDX_agg = pd.concat([NDX_agg, pd.DataFrame(columns = ['IntradayRV']),pd.DataFrame(columns = ['Frequency'])])

for i in range(len(NDX_agg)):
   temp_RV=0
   temp_num=0
   for j in range(1,len(NDX)):
       if NDX_agg.Date[i]==NDX.Date[j] and NDX.Date[j]==NDX.Date[j-1]:
               temp_RV=temp_RV+NDX.log_r[j]
               temp_num=temp_num+1
   NDX_agg.IntradayRV[i]= np.sqrt(temp_RV/temp_num * 87 * 252) 
   NDX_agg.Frequency[i]=temp_num  

NDX_agg = NDX_agg.rename(columns={'IntradayRV': 'NDX_IntradayRV'})
#create new data frame with unique dates, sum up intraday squared returns per day
AAPL_agg = pd.DataFrame(NDX.Date.unique())
AAPL_agg.columns = ['Date']

AAPL_agg = pd.concat([NDX_agg, pd.DataFrame(columns = ['IntradayRV']),pd.DataFrame(columns = ['Frequency'])])

for i in range(len(AAPL_agg)):
   temp_RV=0
   temp_num=0
   for j in range(1,len(AAPL)):
       if AAPL_agg.Date[i]==AAPL.Date[j] and AAPL.Date[j]==AAPL.Date[j-1]:
           temp_RV=temp_RV+AAPL.log_r[j]
           temp_num=temp_num+1
   AAPL_agg.IntradayRV[i]= np.sqrt(temp_RV/temp_num * 87 * 252) 
   AAPL_agg.Frequency[i]=temp_num  
AAPL_agg = AAPL_agg.rename(columns={'IntradayRV': 'AAPL_IntradayRV'})
#
#print(AAPL_agg['IntradayRV'].mean())
#print(AAPL_agg['IntradayRV'].max())
#
#print(NDX_agg['IntradayRV'].mean())
#print(NDX_agg['IntradayRV'].max())

Comb_agg = pd.DataFrame(NDX.Date.unique())
Comb_agg.columns = ['Date']

Comb_agg = pd.concat([Comb_agg, pd.DataFrame(columns = ['IRV_ratio'])])

for i in range(len(Comb_agg)):
   Comb_agg.IRV_ratio[i] = AAPL_agg.AAPL_IntradayRV[i] / NDX_agg.NDX_IntradayRV[i]


#print(Comb_agg.IRV_ratio.mean(),Comb_agg.IRV_ratio.max())
#print(Comb_agg.IRV_ratio.quantile(q=0.9))
#print(Comb_agg.IRV_ratio.quantile(q=0.1))
Comb_agg = pd.concat([Comb_agg, pd.DataFrame(columns = ['AAPL_IntradayRV']),
                 pd.DataFrame(columns = ['NDX_IntradayRV'])])
    
Comb_agg['AAPL_IntradayRV']=AAPL_agg['AAPL_IntradayRV']
Comb_agg['NDX_IntradayRV']=NDX_agg['NDX_IntradayRV']
##############################################
# part III: Merge the Sentiment analysis results 
# and the RV results
##############################################

# create a new data frame to list all dates and store final results 
Datelist=datelist('20180101','20190101')
Datelist=pd.DataFrame(Datelist)
Datelist.columns = ['Date']
for i in range(len(Datelist)):
    Datelist.ix[i,'Date']= datetime.strptime(Datelist.ix[i,'Date'], '%Y-%m-%d').date()


fdf=pd.merge(df_agg,Datelist,how='outer', on='Date')
fdf=fdf.sort_values(['Date'])

# find out the subset of the final dateframe (for matching the finacial data length)
#min_date=datetime.strptime('2018-06-01','%Y-%m-%d').date()
#sub_fdf=fdf.ix[fdf.Date>=min_date]


final_df=pd.merge(fdf,Comb_agg, on='Date')
##############################################
# part IV: Plotting Results
##############################################


#plot the sensitivity (averaged per day)
fig = plt.figure()
plt.plot(df_agg.Date,df_agg.SentScore, 'o-',linewidth=1,markersize=2)

plt.title('Sentiment')
#plt.xlabel('Time')
plt.ylabel('Sentiment')
#plt.gcf().autofmt_xdate()
plt.savefig('Sentiment_Apple.png',dpi = 720,transparent=True)

#plot the dispersion and the quantile 
fig = plt.figure()
plt.plot(df_agg.Date,df_agg.Dispersion, 'o-',linewidth=0, markersize=2)
disp_mean = df_agg.Dispersion.mean()
disp_q10 = df_agg.Dispersion.quantile(q=0.1)
disp_q90 = df_agg.Dispersion.quantile(q=0.9)

plt.axhline(y=disp_mean, xmin=0, xmax=1, linewidth=1, color = 'k')
plt.axhline(y=disp_q10, xmin=0, xmax=1, linewidth=1, color = 'r')
plt.axhline(y=disp_q90, xmin=0, xmax=1, linewidth=1, color = 'r')
plt.title('Dispersion')
plt.savefig('Sentiment_Dispersion_Apple.png',dpi = 720,transparent=True)

#plot the Apple and NDX IRV 
fig = plt.figure()
plt.plot(NDX_agg.Date,NDX_agg.NDX_IntradayRV, 'b',linewidth=1, markersize=2)
plt.plot(AAPL_agg.Date,AAPL_agg.AAPL_IntradayRV, 'r',linewidth=1, markersize=2)

plt.title('Annualized 5min Intervall Intraday Realised Volatility')
#plt.xlabel('Time')
plt.ylabel('Realised Intraday Volatility (%)')
#plt.gcf().autofmt_xdate()
plt.legend(loc='best',frameon=False)
plt.savefig('RealisedIntradayVol.png',dpi = 720,transparent=True)


# plot IRV ratio
fig = plt.figure()
plt.plot(Comb_agg.Date,Comb_agg.IRV_ratio, 'o-',linewidth=1, markersize=2)

IRVratio_mean = Comb_agg.IRV_ratio.mean()
IRVratio_q10 = Comb_agg.IRV_ratio.quantile(q=0.1)
IRVratio_q90 = Comb_agg.IRV_ratio.quantile(q=0.9)

plt.axhline(y=IRVratio_mean, xmin=0, xmax=1, linewidth=1, color = 'k')
plt.axhline(y=IRVratio_q10, xmin=0, xmax=1, linewidth=1, color = 'r')
plt.axhline(y=IRVratio_q90, xmin=0, xmax=1, linewidth=1, color = 'r')

plt.title('Ratio of Intraday Realised Volatility')
#plt.xlabel('Time')
plt.ylabel('Intraday Realised Volatility Ratio')
#plt.gcf().autofmt_xdate()
plt.legend(loc='best',frameon=False )
plt.savefig('RVRatio.png',dpi = 720,transparent=True)




# plot the results of dispersion and IRV Ratio
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(final_df.Date, final_df.Dispersion,'b',linewidth=1, markersize=2)
ax1.set_ylabel('Dispersion')
ax1.set_title("Intraday RV Ratio vs Dispersion")
ax1.legend(loc=0,frameon=False)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(final_df.Date, final_df.IRV_ratio, 'r', linewidth=1, markersize=2)
ax2.set_ylim([0, 3])
ax2.set_ylabel('IRV Ratio')
#ax2.set_xlabel('Date')
ax2.legend(loc='lower right',frameon=False)
plt.savefig('Intraday RV Ratio vs Dispersion.png',dpi = 720,transparent=True)
plt.show()

# plot the results of sentiment and IRV Ratio
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(final_df.Date, final_df.SentScore,'b',linewidth=1, markersize=2)
ax1.set_ylabel('Sentiment')
ax1.set_title("Intraday RV Ratio vs Sentiment")
ax1.legend(loc=0,frameon=False)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(final_df.Date, final_df.IRV_ratio, 'r',linewidth=1, markersize=2)
ax2.set_ylim([0, 3])
ax2.set_ylabel('IRV Ratio')
#ax2.set_xlabel('Date')
ax2.legend(loc='lower right',frameon=False)
plt.savefig('Intraday RV Ratio vs Sentiment.png',dpi = 720,transparent=True)
plt.show()



# plot the results of sentiment and Apple intraday RV
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(final_df.Date, final_df.SentScore,'b',linewidth=1, markersize=2)
ax1.set_ylabel('Sentiment')
ax1.set_title("AAPL Intraday RV  vs Sentiment")
ax1.legend(loc=0,frameon=False)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(final_df.Date, final_df.AAPL_IntradayRV, 'r',linewidth=1, markersize=2)
#ax2.set_ylim([0, 2])
ax2.set_ylabel('IRV')
#ax2.set_xlabel('Date')
ax2.legend(loc='lower right',frameon=False)
plt.savefig('AAPL Intraday RV vs Sentiment.png',dpi = 720,transparent=True)
plt.show()


# plot the results of dispersion and Apple intraday RV
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(final_df.Date, final_df.Dispersion,'b',linewidth=1, markersize=2)
ax1.set_ylabel('Dispersion')
ax1.set_title("AAPL Intraday RV vs Dispersion")
ax1.legend(loc=0,frameon=False)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(final_df.Date, final_df.AAPL_IntradayRV, 'r',linewidth=1, markersize=2)
#ax2.set_ylim([0, 3])
ax2.set_ylabel('IRV')
#ax2.set_xlabel('Date')
ax2.legend(loc='lower right',frameon=False)
plt.savefig('AAPL Intraday RV vs Dispersion.png',dpi = 720,transparent=True)
plt.show()


# plot the results of sentiment and NDX intraday RV
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(final_df.Date, final_df.SentScore,'b',linewidth=1, markersize=2)
ax1.set_ylabel('Sentiment')
ax1.set_title("NDX Intraday RV  vs Sentiment")
ax1.legend(loc=0,frameon=False)
ax2 = ax1.twinx()  # this is the important function
ax2.plot(final_df.Date, final_df.NDX_IntradayRV, 'r',linewidth=1, markersize=2)
#ax2.set_ylim([0, 2])
ax2.set_ylabel('IRV')
#ax2.set_xlabel('Date')
ax2.legend(loc='lower right',frameon=False)
plt.savefig('NDX Intraday RV vs Sentiment.png',dpi = 720,transparent=True)
plt.show()


##############################################
# part V: Check the results
##############################################
print(df_agg.ix[df_agg.SentScore>0.9],df_agg.ix[df_agg.SentScore<-0.9])


