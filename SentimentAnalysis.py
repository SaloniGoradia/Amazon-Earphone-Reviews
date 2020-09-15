#!/usr/bin/env python
# coding: utf-8
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import streamlit as st
import nltk
#nltk.data.path.append('C:/Users/sgoradia/Desktop/Python/SentimentAnalysis/nltk_data')
#nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
#pip install textblob
from textblob import TextBlob
import plotly.graph_objs as go
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')
from wordcloud import WordCloud
from PIL import Image

#Read the csv file
st.title("Sentiment Analysis on Amazon_Earphone_Reviews")
df= pd.read_csv(r"C:/Users/sgoradia/Desktop/Python/Amazon_Earphone reviews.csv")
df1= pd.read_csv(r"C:/Users/sgoradia/Desktop/Python/ProductInfo.csv")
st.header("Glimpse of Amazon_Earphone_Reviews dataset")
st.write(df.head())
st.write("How many rows and columns are there in the data set?")
df.shape
st.header("Products Information")
st.write(df1)
st.write("Dimensions of products information dataset:")
df1.shape

#Info about two csv files
df.info()
df1.info()

#Splitting dataset into 10 datasets based on products
df_p1=df[df.Product=="boAt Rockerz 255"]
df_p2=df[df.Product=="Flybot Wave"]
df_p3=df[df.Product=="Flybot Boom"]
df_p4=df[df.Product=="PTron Intunes"]
df_p5=df[df.Product=="Flybot Beat"]
df_p6=df[df.Product=="Samsung EO-BG950CBEIN"]
df_p7=df[df.Product=="JBL T205BT"]
df_p8=df[df.Product=="Sennheiser CX 6.0BT"]
df_p9=df[df.Product=="Skullcandy S2PGHW-174"]
df_p10=df[df.Product=="JBL T110BT"]
st.header("Reviews for first product (df_p1): boAt Rockerz 255")
st.write(df_p1.head())

#List of all 10 datassets
dblist=[df_p1,df_p2,df_p3,df_p4,df_p5,df_p6,df_p7,df_p8,df_p9,df_p10]

#Adding Polarity scores to df_p1 (reviews for boAt Rockerz 255 )
def scores(x):
    list1=[]
    x.reset_index(inplace=True)
    x.drop("index",axis=1,inplace=True)
    for i in x.ReviewBody:
        list1.append(sia.polarity_scores(i))
    x[["Negative","Neutral","Positive","Compound"]]=pd.DataFrame(list1)
for j in dblist:
    scores(j)
st.header("Adding Polarity scores to df_p1 ")
st.write(df_p1.head(10))

#Positive vs Negative score for df_p1
st.header("Positive vs Negative score for df_p1")
a=px.bar(y="Positive",data_frame=df_p1,width=20000, height=400, color = "Negative",
template = "plotly", title = "Positive vs Negative Reviews")
st.plotly_chart(a)

#Positive vs neutral
st.header("Positive vs Neutral score for df_p1")
b=px.bar(y="Positive",data_frame=df_p1,width=20000, height=400, color = "Neutral",
 template = "plotly", title = "Positive vs Neutral Reviews")
st.plotly_chart(b)

#Compound score for df_p1
st.header("Compound Score for df_p1")
c=px.bar(y="Compound",data_frame=df_p1,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

#Adding class to df_p1
def cls(x):
    list2=[]
    for i in x["Compound"]:
        if i<0:
            list2.append("Negative")
        elif i==0:
            list2.append("Neutral")
        else:
            list2.append("Positive")
    x["Class"]=list2

for j in dblist:
    cls(j)
st.header("Adding class for df_p1")
st.write(df_p1.head(10))

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p1["Class"].value_counts())

colors = ['blue', 'yellow', 'green']

Positive=df_p1[df_p1.Class=="Positive"]["Class"].count()
Negative=df_p1[df_p1.Class=="Negative"]["Class"].count()
Neutral=df_p1[df_p1.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#Finding Keywords in the ReviewBody
def keys(x):
    a = [None] * len(x)
    for i in range(0,(len(x)-1)):
        list3=[]
        blob = TextBlob(x.iloc[i][1])
        for word, tag in blob.tags:
            if (tag=="JJ")| (tag=="VBN")| (tag=="NNS")| (tag=="NN"):
                list3.append(word.lemmatize())
            a[i]=list3
    x["Keywords"]=a
for j in dblist:
    keys(j)
st.header("Adding keywords to df_p1")
st.write(df_p1.head(10))

#Wordcloud for df_p1
list=[]
for i in df_p1["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = "white").generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p1")
st.pyplot(plt)

##Analysis for product 2

st.header("Reviews for product (df_p2): Flybot Wave")
st.write(df_p2.head())
#Compound score for df_p2
st.header("Compound Score for df_p2")
c=px.bar(y="Compound",data_frame=df_p2,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p2["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p2[df_p2.Class=="Positive"]["Class"].count()
Negative=df_p2[df_p2.Class=="Negative"]["Class"].count()
Neutral=df_p2[df_p2.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p2
list=[]
for i in df_p2["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p2")
st.pyplot(plt)

##Analysis for product 3
st.header("Reviews for product (df_p3): Flybot Boom")
st.write(df_p3.head())
#Compound score for df_p3
st.header("Compound Score for df_p3")
c=px.bar(y="Compound",data_frame=df_p3,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p3["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p3[df_p3.Class=="Positive"]["Class"].count()
Negative=df_p3[df_p3.Class=="Negative"]["Class"].count()
Neutral=df_p3[df_p3.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p3
list=[]
for i in df_p3["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p3")
st.pyplot(plt)

##Analysis for product 4
st.header("Reviews for first product (df_p4): PTron Intunes")
st.write(df_p4.head())
#Compound score for df_p4
st.header("Compound Score for df_p4")
c=px.bar(y="Compound",data_frame=df_p4,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p4["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p4[df_p4.Class=="Positive"]["Class"].count()
Negative=df_p4[df_p4.Class=="Negative"]["Class"].count()
Neutral=df_p4[df_p4.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p4
list=[]
for i in df_p4["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p4")
st.pyplot(plt)

##Analysis for product 5
st.header("Reviews for product (df_p5): Flybot Beat")
st.write(df_p5.head())
#Compound score for df_p5
st.header("Compound Score for df_p5")
c=px.bar(y="Compound",data_frame=df_p5,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p5["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p5[df_p5.Class=="Positive"]["Class"].count()
Negative=df_p5[df_p5.Class=="Negative"]["Class"].count()
Neutral=df_p5[df_p5.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p5
list=[]
for i in df_p5["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p5")
st.pyplot(plt)

##Analysis for product 6
st.header("Reviews for product (df_p6): Samsung EO-BG950CBEIN")
st.write(df_p6.head())
#Compound score for df_p6
st.header("Compound Score for df_p6")
c=px.bar(y="Compound",data_frame=df_p6,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p6["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p6[df_p6.Class=="Positive"]["Class"].count()
Negative=df_p6[df_p6.Class=="Negative"]["Class"].count()
Neutral=df_p6[df_p6.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p6
list=[]
for i in df_p6["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p6")
st.pyplot(plt)

##Analysis for product 7
st.header("Reviews for product (df_p7): JBL T205BT")
st.write(df_p7.head())
#Compound score for df_p7
st.header("Compound Score for df_p7")
c=px.bar(y="Compound",data_frame=df_p7,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p7["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p7[df_p7.Class=="Positive"]["Class"].count()
Negative=df_p7[df_p7.Class=="Negative"]["Class"].count()
Neutral=df_p7[df_p7.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p7
list=[]
for i in df_p7["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p7")
st.pyplot(plt)

##Analysis for product 8
st.header("Reviews for product (df_p8): Sennheiser CX 6.0BT")
st.write(df_p8.head())
#Compound score for df_p8
st.header("Compound Score for df_p8")
c=px.bar(y="Compound",data_frame=df_p8,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p8["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p8[df_p8.Class=="Positive"]["Class"].count()
Negative=df_p8[df_p8.Class=="Negative"]["Class"].count()
Neutral=df_p8[df_p8.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p8
list=[]
for i in df_p8["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p8")
st.pyplot(plt)

##Analysis for product 9
st.header("Reviews for product (df_p9):Skullcandy S2PGHW-174")
st.write(df_p9.head())
#Compound score for df_p9
st.header("Compound Score for df_p9")
c=px.bar(y="Compound",data_frame=df_p9,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p9["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p9[df_p9.Class=="Positive"]["Class"].count()
Negative=df_p9[df_p9.Class=="Negative"]["Class"].count()
Neutral=df_p9[df_p9.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p9
list=[]
for i in df_p9["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p9")
st.pyplot(plt)

##Analysis for product 10
st.header("Reviews for first product (df_p10): JBL T110BT")
st.write(df_p10.head())
#Compound score for df_p10
st.header("Compound Score for df_p10")
c=px.bar(y="Compound",data_frame=df_p10,width=20000, height=400, template = "ggplot2", title = "compoundscore")
st.plotly_chart(c)

st.header("How many reviews are Positive, Negative and Neutral?")
st.write(df_p10["Class"].value_counts())

#Pi chart
colors = ['blue', 'yellow', 'green']

Positive=df_p10[df_p10.Class=="Positive"]["Class"].count()
Negative=df_p10[df_p10.Class=="Negative"]["Class"].count()
Neutral=df_p10[df_p10.Class=="Neutral"]["Class"].count()
fig = go.Figure(data=[go.Pie(labels=['PositiveComments','NeutralComments','NegativeComments'],
                             values=[Positive, Neutral, Negative])])
fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
st.header("Pi chart of Positive, Negative and Neutral comments")
st.plotly_chart(fig)

#WordCloud for df_p10
list=[]
for i in df_p10["Keywords"] :
    list.append(i)
slist = str(list)

wordcloud = WordCloud(width=1000, height=500, background_color = 'white').generate(slist)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
st.header("WordCloud for df_p10")
st.pyplot(plt)
