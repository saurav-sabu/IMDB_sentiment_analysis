import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


nltk.download('punkt')
nltk.download('stopwords')

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("Total_10000_records.csv")


model_bnb = pickle.load(open("model_cv_bnb.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

ps = PorterStemmer()

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and "}


def processing_data(text):
    
    text = text.lower()
    
    for key in contractions.keys():
            value = contractions[key]
            text = text.replace(key,value)
            
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
    
    text = re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if (i not in stopwords.words("english")) and (i not in string.punctuation ):
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


nav = st.sidebar.radio(
    "Navigation", ["About The Project", "Dataset Description","Make Prediction", "Key Insights"])

if nav == "About The Project":

    st.markdown("# About The Project")

    st.image("imdb.jpg")

    st.markdown("---")
    st.markdown("""
IMDb (an acronym for Internet Movie Database) is an online database of information related to films, television series, podcasts, home videos, video games, and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews. IMDb began as a fan-operated movie database on the Usenet group "rec.arts.movies" in 1990, and moved to the Web in 1993. Since 1998, it has been owned and operated by IMDb.com, Inc., a subsidiary of Amazon.
    """,unsafe_allow_html=True)

    st.markdown('''
`Sentiment analysis of movies plays a crucial role in several aspects of the film industry and beyond. Here are some of the key reasons why it is important:`

1. **Audience Insights:** Understanding the sentiments and emotions of the audience is vital for filmmakers and studios. Sentiment analysis helps them gauge how well a movie is likely to be received, which can inform marketing strategies and potentially influence the direction of future projects.

2. **Box Office Predictions:** Sentiment analysis can be used to predict the box office performance of a movie. By analyzing social media chatter and online reviews, industry experts and studios can estimate a film's potential success, allowing for better resource allocation and marketing campaigns.

3. **Content Improvement:** Filmmakers and studios can use sentiment analysis to identify specific aspects of a movie that audiences love or dislike. This information can be used to improve the content and make necessary adjustments in real-time or for future projects.

4. **Marketing and Promotion:** Sentiment analysis can guide marketing and promotional efforts. By understanding what elements of the movie resonate most with the target audience, marketing campaigns can be tailored to highlight those aspects and attract more viewers.
''')

elif nav == "Dataset Description":

    st.markdown("# About the Dataset")

    st.markdown('''
**Dataset Name:** IMDb Movie Review Dataset

**Description:**
The IMDb Movie Review Dataset is a collection of movie reviews and associated sentiment labels, compiled from the Internet Movie Database (IMDb). It is commonly used for sentiment analysis, natural language processing (NLP), and text classification tasks. This dataset provides valuable insights into the sentiments and opinions of viewers towards various movies.

**Key Information:**

Data Source: The dataset is obtained from user-generated reviews on the IMDb platform, which contains a wide range of movies from different genres and time periods.

Size: The dataset consists of a large number of movie reviews, with a typical dataset size ranging from tens of thousands to millions of reviews, depending on the specific version or subset of the dataset used.

Data Format: Each entry in the dataset typically includes two main components:

Textual Review: The full-text review written by an IMDb user, providing their thoughts, opinions, and comments about a particular movie.
Sentiment Label: A binary sentiment label that categorizes the review as either "positive" or "negative." This label is often determined based on the user's rating or review text.
Sentiment Labeling: Sentiment labels are assigned based on the user-provided ratings or, in some cases, determined using text analysis algorithms applied to the review text. A "positive" sentiment label typically corresponds to a high rating (e.g., 7-10 stars), while a "negative" sentiment label corresponds to a low rating (e.g., 1-4 stars).
''')
    

elif nav == "Make Prediction":

    st.title("Sentiment Analysis")

    text = st.text_area("Enter review of your favourite movie:")

    if st.button("Submit"):
        
        text = processing_data(text)

        text = cv.transform([text]).toarray()

        if model_bnb.predict(text) == 1:
            st.write("POSITIVE")
        else:
            st.write("NEGATIVE")
        
elif nav == "Key Insights":
    
    st.title("Important Analysis")

    st.markdown("---")

    wc = WordCloud(width = 500,height=500,min_font_size=10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positive Words:")
        positive_wc = wc.generate(df[df["sentiment"] == 1]["review"].str.cat(sep=" "))
        plt.imshow(positive_wc)
        st.pyplot()

    with col2:
        st.subheader("Negative Words:")
        negative_wc = wc.generate(df[df["sentiment"] == 0]["review"].str.cat(sep=" "))
        plt.imshow(negative_wc)
        st.pyplot()

    col3, col4  = st.columns(2)
    
    with col3:
        st.subheader("Distribution of Number of Characters")
        sns.histplot(x="num_chars",data=df,hue="sentiment")
        st.pyplot()

    with col4:
        st.subheader("Distribution of Number of Words")
        sns.histplot(x="num_words",data=df,hue="sentiment")
        st.pyplot()

    with st.container():
        st.subheader("Distribution of Polarity")
        df["polarity"].plot(kind="hist",bins=50)
        plt.xlabel("polarity")
        plt.ylabel("count")
        plt.title("Sentiment polarity Distribution")
        st.pyplot()



