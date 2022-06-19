#!/usr/bin/env python
# coding: utf-8

# # Création de l'Application Web
# 
# L'idée est de produire une page web avec des représentations graphiques des données textes (via Streamlit).
# 
#     Pour lancer l'application Streamlit:
# **streamlit hello**
#     
#     Pour lancer la page Web Streamlit
# **streamlit run C:\Users\blanc\OpenClassrooms\IA_Project6_Openclassrooms_IAstart-up\IA_Start_Up_Webpage.py**
# 
# **streamlit run ~\IA_Start_Up_Webpage.py**
# 

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st
from PIL import Image
import re


#Librairie de Tokenisation
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.tag.util import str2tuple

from joblib import dump, load
import pickle


#import pyLDAvis
#import pyLDAvis.sklearn

#from gensim import corpora
#import pyLDAvis.gensim
#import gensim
#from gensim.models import CoherenceModel



# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import Model, preprocessing, regularizers
from keras.applications import VGG16, EfficientNetB0
from keras.applications.vgg16 import preprocess_input, decode_predictions
#from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as p_img
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn import cluster, metrics, preprocessing, manifold, decomposition
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.metrics.cluster import adjusted_rand_score


# In[3]:


st.title('IA Project 6 - WebPage to manage Text & Image data')


# ## Chargement des modèles
# 
# Comme nous l'avons fait lors de l'analyse des données et des modèles, nous utilisons la **libraire Joblib** pour ré-utiliser nos modèls à partir de nouvelles données acquises depuis l'API.

# In[4]:


#Chargement du modèle d'encodage tfidf entrainé précédemment? #tfidf = TfidfVectorizer(min_df=0.005,max_df=0.8)
tfidf = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/tfidfvectorizer.joblib')

# Chargement  du modèle LDA   #lda_tfidf.fit(values_tfidf)
lda_tfidf_api = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/lda_tfidf.joblib')

#Chargement du modèle d'encodage tfidf entrainé précédemment? #tfidf = TfidfVectorizer(min_df=0.005,max_df=0.8)
#tfidf_gensim = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/tfidf_gensim.joblib')

# Chargement  du modèle LDA  GENSIM
#lda_gensim_api = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/lda_gensim.joblib'


# ## Creation d'un pipeline

# In[5]:


def custom_nlp_pipeline(data: pd.DataFrame, max_score_filter=2, rating_column='rating', text_column='text', tag_type_to_eliminate=['NN','RB','RBR','IN','VB','VBN','VBG','VBZ','MD','CD','PRP','PRP$']):
    #filter on unstasfied comments (rating under max_score_filter variable)
    data = data.loc[(data[rating_column] <= max_score_filter), :].copy()
    
    # Change Text column to lower case
    data[text_column] = data.loc[:,text_column].str.lower()
    
    # Replace punct by blank
    data[text_column] = data.loc[:,text_column].str.replace(r'[^\w\s]+', '', regex=True)
    #data[text_column] = data[text_column].str.replace(r'[^\w\s]+', '', regex=True)
    
    # Remove numbers digits from text column
    data['text_only'] = data.loc[:,text_column].str.replace(r'[\d]+', '', regex=True) 
    #data['text_only'] = data[text_column].str.replace('\d+', '', regex=True)
    
    # Words splitting
    data = tokenize_text(data, 'text_only')
    #Remove all unecessary tagged words (Adverbs, Verbs, ...)
    data = remove_undesired_wordtag(data, 'tokenized_text', tag_type_to_eliminate)
    
    # Words cleaning
    data = stopwords_text(data, 'removed_tag_text')
    data = lemmatize_text(data, 'stopwords_text')
    
    #Création du bag of words
    data['thesaurus'] = data.loc[:,'lemmatized_text'].apply(lambda x: " ".join(x))
    
    #Extraction des features avec Tf-Idf Vectorizer : min_df=0.01,max_df=0.8 (précedemment entrainé, et chargé depuis Joblib précedemment)
    values = tfidf.transform(data['thesaurus'])
    #values = tfidf.fit_transform(data['thesaurus'])
    #print("Created %d X %d TF-IDF-normalized document-term matrix" % (values.shape[0], values.shape[1]) )
    
    #From the thesaurus text, split each word and create new dataframe counting word presence frequency
    word_frequency = data.thesaurus.str.split(expand=True).stack().value_counts()
    word_frequency = pd.DataFrame({'word':word_frequency.index, 'word_count':word_frequency.values})
    #For each word, determne its TAG type
    word_frequency['word_tag'] = word_frequency['word'].apply(lambda x: nltk.pos_tag([x])[0][1])
    
    return(data, values, word_frequency)

# Init the Stopword language
stop = stopwords.words('english')

#Function to remove useless words 
def stopwords_text(df, column):
    #df['stopwords_text'] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['stopwords_text'] = df.loc[:,column].apply(lambda x: [w for w in x if not w in stop and w.isalpha()])
    return (df)

# Init the Wordnet Lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

#Function to lemmatize text (Generic representation of words)
def lemmatize_text(df, column):
    #df['lemmatized_text'] = df.apply(lambda row: lemmatizer.lemmatize(row[column]), axis=1)
    df['lemmatized_text'] = df[column].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    #df['lemmatized_text'] = df.loc[:,column].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    return (df)

# Init the Tokenizer
tokenizer = nltk.RegexpTokenizer(r'\w+')

#Function to tokenize text (split in array)
def tokenize_text(df, column):
    df['tokenized_text'] = df.apply(lambda row: tokenizer.tokenize(row[column]), axis=1)
    #df['tokenized_text'] = df.apply(lambda row: tokenizer.tokenize(row[column]))
    return (df)

def remove_undesired_wordtag(df, column, tag_type_to_eliminate):
    #df['removed_tag_text'] = df.apply(lambda row: pos_tag(row[column]), axis=1)
    df['removed_tag_text'] = df.apply(lambda row: ([s[0] for s in pos_tag(row[column]) if s[1] not in tag_type_to_eliminate]), axis=1)
    return df

def load_data(DATA_URL, nrows):
    data_loaded = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data_loaded.rename(lowercase, axis='columns', inplace=True)
    return (data_loaded)

def custom_nlp_pipeline2(text_review, tag_type_to_eliminate=['NN','RB','RBR','IN','VB','VBN','VBG','VBZ','MD','CD','PRP','PRP$']):
    
    # Change Text column to lower case
    text_review_processed = text_review.lower()
    
    # Replace punct by blank
    text_review_processed = re.sub(r'[^\w\s]','',text_review_processed)
    #text_review_processed = text_review_processed.replace(r'[^\w\s]+', '', regex=True)
    
    # Remove numbers digits from text column
    #text_review_processed = re.sub(r'[^\d]','',text_review_processed)
    #text_review_processed = text_review_processed.replace(r'[\d]+', '', regex=True) 
    
    # Words splitting
    text_review_processed = word_tokenize(text_review_processed)
    
    #text_review_processed = [s for s in text_review_processed if s[1] not in tag_type_to_eliminate]
    text_review_processed = [s[0] for s in pos_tag(text_review_processed) if s[1] not in tag_type_to_eliminate]
    
    # Words cleaning
    text_review_processed = [word for word in text_review_processed if word not in stopwords.words('english')]

    #text_review_processed = lemmatizer.lemmatize(text_review_processed)
    text_review_processed = ' '.join([lemmatizer.lemmatize(w) for w in text_review_processed])
     #Création du bag of words
    #text_review_processed = text_review_processed.apply(lambda x: " ".join(x))
    
    text_review_processed = pd.Series(text_review_processed)
    #d = {'thesaurus': text_review_processed}
    #text_review_processed = pd.DataFrame(data=d)
    
    #Extraction des features avec Tf-Idf Vectorizer : min_df=0.01,max_df=0.8 (précedemment entrainé, et chargé depuis Joblib précedemment)
    values = tfidf.transform(text_review_processed)
    
    return(text_review_processed, values)


# ## Chargement du fichier CSV
# 

# In[6]:


image = Image.open('C:/Users/blanc/Documents/ConnectERPrise_Logo.jpg')
with st.sidebar:
    st.image(image, caption='My Logo')


# ### Text Processing

# In[7]:


st.header('Text Data Processing - From Existing CSV')
st.subheader(f'Text Data Extract Process')


# In[8]:


#Choose how many rows to import from CSV
nrows = st.slider('How many reviews do you want to import from CSV?', 1, 10000, 5000)
st.write("You selected ", nrows, 'number of reviews')


# In[9]:


uploaded_file = st.file_uploader("Choose a CSV file corresponding to API GraphQL extract performed previously")
if uploaded_file is not None:
        # Create a text element and let the reader know the data is loading.
        data_load_state = st.text('Loading data...')
        # Load n rows of data into the dataframe.
        data_imported_csv = load_data(uploaded_file, nrows=nrows)
        # Notify the reader that the data was successfully loaded.
        data_load_state.text('Loading data...done!')
        #st.write(dataframe)
        st.write(f'Imported {nrows} rows from **CSV file {uploaded_file.name}**, are stored in Dataframe')
        if st.checkbox('Show Imported dataframe'):
            st.dataframe(data_imported_csv)
        #st.write(data)


# In[10]:


#st.subheader(f'Imported {nrows} rows from CSV file, are stored in Dataframe')
#st.write(data)


# In[11]:


st.subheader(f'Text Data Transform Process')


# In[12]:


#Define variables
max_score_filter=2
rating_column='rating'
text_column='text'
#tag_type_to_eliminate = ['NN','RB','RBR','IN','VB','VBN','VBG','VBZ','MD','CD','PRP','PRP$']
tag_type_to_eliminate = ['RB','RBR','MD','CD']

if uploaded_file is not None:
    #Rune the pipeline on df_api data
    df_api_pipeline, values, word_frequency = custom_nlp_pipeline(pd.DataFrame(data_imported_csv), max_score_filter, rating_column, text_column, tag_type_to_eliminate)
    st.write('Dataframe is filtered on negative evaluation, and associated reviews text are preprocessed : lowercase + punct + digits + tokenisation + stopwords + lemmatization + bag of words')
    if st.checkbox('Show dataframe at Transformation Step 1 : NLP'):
        st.dataframe(df_api_pipeline)


# In[13]:


def simple_topic_name(row):
    if row['Topic'] == 0:
        val = 'Nourriture, Aliments et Gouts'
    elif row['Topic'] == 1:
        val = 'Réservation'
    elif row['Topic'] == 2:
        val = 'Bar et boissons'
    else:
        val = 'Temps d attente et Service'
    return val


# In[14]:


n_topics = 4

if uploaded_file is not None:
    values_tfidf = pd.DataFrame(values.toarray())
    topic_result_csv_df = pd.DataFrame(lda_tfidf_api.transform(values_tfidf)).idxmax(axis=1)
    #values_tfidf.head()

    df_api_results = pd.concat([df_api_pipeline.loc[:, ['rating', 'text', 'name']].reset_index(drop=True), pd.DataFrame(lda_tfidf_api.transform(values_tfidf)).reset_index(drop=True)], axis = 1)
    df_api_results['Topic'] = topic_result_csv_df #pd.DataFrame(lda_tfidf_api.transform(values_tfidf)).idxmax(axis=1)
    df_api_results['Topic_name'] = df_api_results.apply(simple_topic_name, axis=1)
    if st.checkbox('Show dataframe at Transformation Step 2 : Topics Detection'):
        #st.dataframe(df_api_results.style.highlight_max(axis=0))
        st.dataframe(df_api_results)


# In[15]:


st.subheader(f'Text Data Visualization Process')


# In[16]:


st.markdown('Here below, we show the imported **CSV file post-processing results**, and more specifically the _*number of reviews for each topic_*.')


# In[17]:


#chart_data = df_api_results['Topic'] #df_api_results.loc[:, ['rating','topic']]
if uploaded_file is not None:
    chart_data = pd.DataFrame(df_api_results['Topic_name'].value_counts(dropna=True, sort=True))

    st.bar_chart(chart_data)

    with st.expander("See explanation"):
        st.write(f"The chart above shows the {n_topics} Topics used to classify text reviews from Imported Data")
        st.write("The dataframe below lists Topics proportion")
        st.dataframe(chart_data)
        
        # Create and generate a word cloud image:
        st.write("The picture below will show top50 most represented words in selected tag from negative evaluation")
        
        options = st.multiselect('What tag do you want to visualise ?',
                                 ['NN','JJ','VBN', 'VBG', 'VB', 'VBD', 'IN', 'NNS'], #['noun','adjective','verb past participle', 'verb present participle', 'verb, base form', 'verb, past tense', 'preposition or conjunction', 'noun, common, plural'],
                                 ['NN','JJ','IN']
                                 )
        st.write('You selected:', options)
        
        #data_wordcloud = word_frequency.loc[word_frequency['word_tag'] == 'JJ', :].set_index('word').to_dict()['word_count']
        data_wordcloud = word_frequency.loc[word_frequency['word_tag'].isin(options) == True, :].set_index('word').to_dict()['word_count']
        wordcloud = WordCloud(max_words = 50, width=800, height=400).generate_from_frequencies(data_wordcloud)
        # Display the generated image:
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


# In[ ]:





# In[18]:


st.header('Data Processing - From New Manual Entry')


# In[19]:


st.subheader(f'Data Extract & Transform & Visualization Process')


# In[20]:


default_value_goes_here = 'Write text review here'
user_input = st.text_input(default_value_goes_here, default_value_goes_here)
st.write('The text pre-processing and classification will now start based on your input', user_input)


# In[21]:


def topic_name(topic_result):
    if topic_result == 0:
        val = 'Nourriture, Aliments et Gouts'
    elif topic_result == 1:
        val = 'Réservation'
    elif topic_result == 2:
        val = 'Bar et boissons'
    else:
        val = 'Temps d attente et Service'
    return val


# In[22]:


if st.button('Press the button to launch the text processing phases'):
    text_review_processed, values = custom_nlp_pipeline2(user_input, tag_type_to_eliminate)
    st.write('the text after post-processing is : ', text_review_processed[0])
    
    #st.write('the post-processing text is associated to topic')
    values_tfidf = pd.DataFrame(values.toarray())
    topic_result = pd.DataFrame(lda_tfidf_api.transform(values_tfidf)).idxmax(axis=1)
    st.write('The text processed is allocated to topic : ', topic_result[0], topic_name(topic_result[0]))
else:
    st.write('Text processing not started')


#user_output = st.write('the post-processing text is', text_review_processed)
#user_output
#topic_assignation = st.write('the post-processing text is associated to topic', values)
#topic_assignation


# In[ ]:





# ### Image processing

# In[ ]:


st.header('Images Processing - From New Manual Entry')
st.subheader(f'Image Extract & Transform & Visualization Process')


# In[5]:


#load kmeans model in joblib files
kmeans_VGG = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/kmeans_VGG.joblib')

tsne_VGG = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/tsne_VGG.joblib')

#load stdscaler used to transform data
std_scale_VGG = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/std_scale_VGG.joblib')

#load kmeans model after pca reduction in joblib files
kmeans_VGG_reduit = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/kmeans_VGG_réduit.joblib')

tsne_VGG_reduit = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/tsne_VGG_réduit.joblib')

#load stdscaler after pca reduction used to transform data
std_scale_VGG_réduit = load('C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/Models/std_scale_VGG_réduit.joblib')


# In[2]:


def label_matrix(label_predicted):
    if label_predicted == 0:
        label_name = 'outside_inside_predicted'
    elif label_predicted == 1:
        label_name = 'inside_outside_predicted'
    elif label_predicted == 2:
        label_name = 'menu_predicted'
    elif label_predicted == 3:
        label_name = 'food_predicted'
    elif label_predicted == 4:
        label_name = 'drink_predicted'
    return label_name



def simple_label_name(row):
    if row['label_predicted'] == 0:
        label_predicted_name = 'outside'
    elif row['label_predicted'] == 1:
        label_predicted_name = 'inside'
    elif row['label_predicted'] == 2:
        label_predicted_name = 'menu'
    elif row['label_predicted'] == 3:
        label_predicted_name = 'food'
    elif row['label_predicted'] == 4:
        label_predicted_name = 'drink'
    return label_predicted_name


# In[ ]:


list_image = ['drink', 'menu', 'inside', 'outside', 'food']

default_path_goes_here = 'C:/Users/blanc/OpenClassrooms/IA_Project6_Openclassrooms_IAstart-up/dataset/photos/'
folder_path = st.text_input('Write Image path folder here', default_path_goes_here)
st.write('The image folder path is : ', folder_path, '   !!! WARNING: This path shall be accurate !!!')


# In[7]:




uploaded_image_file = st.file_uploader(f"Choose a Image file corresponding to one of following Category : {list_image}", type=("jpg","png","jpeg"))
if uploaded_image_file is not None:
        # Create a text element and let the reader know the data is loading.
        image_load_state = st.text('Loading image...')
        # Load image
        image_imported = Image.open(uploaded_image_file)
        image_path = folder_path + uploaded_image_file.name
        st.write(f'Imported file path is {image_path}')
        

        
        # Notify the reader that the data was successfully loaded.
        image_load_state.text('Loading image...done!')
        #st.write(dataframe)
        st.write(f'Image has been successfully imported')
        


# In[8]:


model = VGG16(weights="imagenet",include_top=False,pooling="avg")
VGG_features = []


if uploaded_image_file is not None:    
    # load an image from file
    img = load_img(image_path, target_size=(224, 224))  # Charger l'image

    # convert the image pixels to a numpy array
    img = img_to_array(img)  # Convertir en tableau numpy
    
    # reshape data for the model
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    
    # prepare the image for the VGG model
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    # Extracting our features
    features = model.predict(img)

    VGG_features.append(features[0])
        
    #Standard scaler on img
    df_VGG_features = pd.DataFrame(VGG_features)
    VGG_features_scaled = std_scale_VGG.transform(df_VGG_features)
    df_VGG_features_scaled = pd.DataFrame(VGG_features_scaled)

    # Extracting our features
    label_predicted = kmeans_VGG.predict(df_VGG_features_scaled)
    st.write(f"Image classification is  : {label_matrix(label_predicted)}")
    
    if st.checkbox('Show Imported Image'):
            st.image(image_imported, caption=label_matrix(label_predicted), width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        #st.write(data)
    
    if st.checkbox('Show image detected features using VGG model'):
        #st.dataframe(df_api_results.style.highlight_max(axis=0))
        st.write("df_VGG_features")
        st.dataframe(df_VGG_features)
        st.write("df_VGG_features_scaled")
        st.dataframe(df_VGG_features_scaled)
        st.write("VGG_features_scaled")
        st.write(VGG_features_scaled)


#     if st.checkbox('Show T-SNE visualisation of predicted class'):
#         df_VGG_features_scaled = st.file_uploader("Choose a df_VGG_features_scaled.csv file")
#         if df_VGG_features_scaled is not None:
#             df_VGG_features_scaled_csv = pd.read_csv(df_VGG_features_scaled)
#             st.write(f'Imported {df_VGG_features_scaled.name} in Dataframe')
#             if st.checkbox('Show Imported df_VGG_features_scaled dataframe'):
#                 st.dataframe(df_VGG_features_scaled_csv)
                
# #             df_tsne_VGG_csv['label_predicted_name'] = df_tsne_VGG_csv.apply(simple_label_name, axis=1)
# #             df_tsne_VGG_imported_image = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
# #             df_tsne_VGG_imported_image["label"] = np.nan
# #             df_tsne_VGG_imported_image["label_predicted"] = label_predicted
# #             df_tsne_VGG_imported_image["label_predicted_name"] = label_matrix(label_predicted)
# #             df_tsne_VGG_to_display = pd.concat([df_tsne_VGG_csv, df_tsne_VGG_imported_image]) 
            
#             df_VGG_features_scaled_concat = pd.concat([df_VGG_features_scaled_csv, df_VGG_features_scaled]) 
#             X_tsne = tsne_VGG.fit_transform(df_VGG_features_scaled_concat) #ou df_bow_VGG ou weighted_images_df['VGG_features'] ou VGG_features_scaled ou df_VGG_features_scaled
#             df_tsne_VGG = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
#             df_tsne_VGG["label"] = weighted_images_df['label']
#             df_tsne_VGG.head()

#             # Display the generated T-SNE
#             #Create numpy array for the visualisation
#             x = df_tsne_VGG_to_display["tsne1"]
#             y = df_tsne_VGG_to_display["tsne2"] 

#             fig = plt.figure(figsize=(10, 10))
#             plt.scatter(x, y)
#             #plt.scatter(x, y, label="label_predicted_name")

#             st.balloons()
#             st.pyplot(fig)
                                       


# In[ ]:




