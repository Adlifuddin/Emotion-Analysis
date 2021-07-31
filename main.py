import pandas as pd 
import numpy as np
import nltk
import streamlit as st
import json
import streamlit_wordcloud as wordcloud
import joblib
import altair as alt
import matplotlib.pyplot as plt
from retrieve import retriever, download_csv
from preprocessing import text_preprocessing

st.set_page_config(layout="wide")

# get data file
with open('datasets/emotion_twitter_data.json') as fopen:
    myfile = json.load(fopen)

data_anger = retriever(myfile,'Text','anger')
data_anger['Emotion'] = 0

data_fear = retriever(myfile,'Text','fear')
data_fear['Emotion'] = 1

data_happy = retriever(myfile,'Text','happy')
data_happy['Emotion'] = 2

data_love = retriever(myfile,'Text','love')
data_love['Emotion'] = 3

data_sadness = retriever(myfile,'Text','sadness')
data_sadness['Emotion'] = 4

data_surprise = retriever(myfile,'Text','surprise')
data_surprise['Emotion'] = 5

# Combine both dataframes into one master dataframe
data = pd.concat([data_anger, data_fear, data_happy, data_love, data_sadness, data_surprise], ignore_index = True)

menu_selectbox = st.sidebar.selectbox(
    "Menu",
    ("Data Visualization", "Emotion Analysis by Text", "Emotion Analysis by File")
)

st.title("Emotion Analysis (Bahasa Melayu)")

if menu_selectbox == 'Data Visualization':
    col1, col2 = st.beta_columns(2)

    with col1:
        st.header("50 Most Mentioned Words")

        words = [
            dict(text="happy", value=22757, color="#b5de2b"),
            dict(text="bodoh", value=20963, color="#b5de2b"),
            dict(text="sakit", value=19251, color="#b5de2b"),
            dict(text="takut", value=18280, color="#b5de2b"),
            dict(text="hati", value=15708, color="#b5de2b"),
            dict(text="kecewa", value=14304, color="#b5de2b"),
            dict(text="malas", value=12463, color="#b5de2b"),
            dict(text="komunis", value=11711, color="#b5de2b"),
            dict(text="mati", value=11681, color="#b5de2b"),
            dict(text="rindu", value=10302, color="#b5de2b"),
            dict(text="suka", value=10252, color="#b5de2b"),
            dict(text="kejut", value=10140, color="#b5de2b"),
            dict(text="tinggal", value=9795, color="#b5de2b"),
            dict(text="sayang", value=9316, color="#b5de2b"),
            dict(text="cinta", value=9047, color="#b5de2b"),
            dict(text="sedih", value=8195, color="#b5de2b"),
            dict(text="marah", value=8163, color="#b5de2b"),
            dict(text="kes", value=7949, color="#b5de2b"),
            dict(text="tengok", value=7867, color="#b5de2b"),
            dict(text="benci", value=7614, color="#b5de2b"),
            dict(text="kapitalis", value=7401, color="#b5de2b"),
            dict(text="jatuh", value=7400, color="#b5de2b"),
            dict(text="amp", value=7327, color="#b5de2b"),
            dict(text="muka", value=7180, color="#b5de2b"),
            dict(text="dukacita", value=7098, color="#b5de2b"),
            dict(text="pergi", value=6880, color="#b5de2b"),
            dict(text="cakap", value=6707, color="#b5de2b"),
            dict(text="rumah", value=6695, color="#b5de2b"),
            dict(text="pasal", value=6545, color="#b5de2b"),
            dict(text="makan", value=6372, color="#b5de2b"),
            dict(text="benda", value=6359, color="#b5de2b"),
            dict(text="tolong", value=6190, color="#b5de2b"),
            dict(text="lelaki", value=6096, color="#b5de2b"),
            dict(text="hidup", value=6061, color="#b5de2b"),
            dict(text="jalan", value=6050, color="#b5de2b"),
            dict(text="dunia", value=6015, color="#b5de2b"),
            dict(text="masuk", value=5981, color="#b5de2b"),
            dict(text="anak", value=5567, color="#b5de2b"),
            dict(text="babi", value=5493, color="#b5de2b"),
            dict(text="harap", value=5389, color="#b5de2b"),
            dict(text="gila", value=5286, color="#b5de2b"),
            dict(text="malaysia", value=5170, color="#b5de2b"),
            dict(text="maklum", value=4938, color="#b5de2b"),
            dict(text="kali", value=4935, color="#b5de2b"),
            dict(text="ngeri", value=4879, color="#b5de2b"),
            dict(text="korang", value=4752, color="#b5de2b"),
            dict(text="takde", value=4569, color="#b5de2b"),
            dict(text="duduk", value=4491, color="#b5de2b"),
            dict(text="jumpa", value=4444, color="#b5de2b"),
            dict(text="udah", value=4435, color="#b5de2b"),
        ]
        return_obj = wordcloud.visualize(words, tooltip_data_fields={'text':'Word', 'value':'Counts'}, per_word_coloring=False)

    # col1, col2,col3 = st.beta_columns(3)

    with col2:
        st.header("Tweets Breakdown by Emotion")

        labels = 'Marah', 'Takut', 'Gembira', 'Cinta', 'Sedih', 'Terkejut'
        sizes = [108813, 20316, 30962, 20783, 26468, 13107]
        explode = (0, 0, 0.1, 0, 0, 0)

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')

        st.pyplot(fig1)


    st.header("Classifier Models Used")

    col1, col2, col3, col4 = st.beta_columns(4)

    with col1:
        st.subheader("Support Vector Classification")
        st.write(f"Accuracy: 98.13 %")
        st.write(f"Precision: 97.78 %")
        st.write(f"Recall: 97.78 %")
        st.write(f"F1-Score: 97.78 %")

    with col2:
        st.subheader("Xgboost")
        st.write(f"Accuracy: 98.53 %")
        st.write(f"Precision: 98.14 %")
        st.write(f"Recall: 98.22 %")
        st.write(f"F1-Score: 98.18 %")

    with col3:
        st.subheader("Logistic Regression")
        st.write(f"Accuracy: 97.90 %")
        st.write(f"Precision: 97.55 %")
        st.write(f"Recall: 97.40 %")
        st.write(f"F1-Score: 97.47 %")

    with col4:
        st.subheader("Stochastic Gradient Descent")
        st.write(f"Accuracy: 97.08 %")
        st.write(f"Precision: 96.63 %")
        st.write(f"Recall: 96.54 %")
        st.write(f"F1-Score: 96.58 %")

elif menu_selectbox == 'Emotion Analysis by Text':

    model_option = st.selectbox(
        "Model Selection",
        ('Logistic Regression', 'Support Vector Classification', 'Xgboost', 'Stochastic Gradient Descent')
    )

    text = st.text_input('Enter text to be analyzed',max_chars=150)

    if st.button('Analyze Text Emotion'):

        # load model
        joblib_XGB_model = joblib.load('joblib_XGB_Model.pkl')
        joblib_SVM_model = joblib.load('joblib_SVM_Model.pkl')
        joblib_LR_model = joblib.load('joblib_LR_Model.pkl')
        joblib_SGD_model = joblib.load('joblib_SGD_Model.pkl')
        
        st.write(f'Text before preprocessing: {text}')

        # clean text
        clean_text = text_preprocessing(str(text))
        st.write(f'Text after preprocessing: {clean_text}')

        # Predict
        def start_pred(pred_data):
            if pred_data[0] == 0:
                st.write('Emotion: Marah')
            elif pred_data[0] == 1:
                st.write('Emotion: Takut')
            elif pred_data[0] == 2:
                st.write('Emotion: Gembira')
            elif pred_data[0] == 3:
                st.write('Emotion: Cinta')
            elif pred_data[0] == 4:
                st.write('Emotion: Sedih')
            elif pred_data[0] == 5:
                st.write('Emotion: Terkejut')

        if model_option == 'Xgboost':
            pred_data=joblib_XGB_model.predict([clean_text])
            start_pred(pred_data)
        elif model_option == 'Logistic Regression':
            pred_data=joblib_LR_model.predict([clean_text])
            start_pred(pred_data)
        elif model_option == 'Support Vector Classification':
            pred_data=joblib_SVM_model.predict([clean_text])
            start_pred(pred_data)
        elif model_option == 'Stochastic Gradient Descent':
            pred_data=joblib_SGD_model.predict([clean_text])
            start_pred(pred_data)

else:
    start_analysis = False

    model_option = st.selectbox(
        "Model Selection",
        ('Logistic Regression', 'Support Vector Classification', 'Xgboost', 'Stochastic Gradient Descent')
    )

    upload_file = st.file_uploader("Upload file", type=["csv"])

    # if st.button('Analyze Text Emotion'):

    if upload_file is not None:
        upload_data = pd.read_csv(upload_file)
        del upload_file

        # load model
        joblib_XGB_model = joblib.load('joblib_XGB_Model.pkl')
        joblib_SVM_model = joblib.load('joblib_SVM_Model.pkl')
        joblib_LR_model = joblib.load('joblib_LR_Model.pkl')
        joblib_SGD_model = joblib.load('joblib_SGD_Model.pkl')

        st.subheader("Initial Data")
        st.write(upload_data)
        st.write('Length of data:', len(upload_data))
        column = list(upload_data.columns)

        if len(column) > 1:
            st.warning("*Warning:* Only one column of text is allowed to be analyzed")
            column_select = st.selectbox(
                "Choose a column to be analyzed",
                (column)
            )

            if st.button('Select'):
                for col in column:
                    if col != column_select:
                        upload_data = upload_data.drop([col], axis=1) 
                
                st.subheader("Data after column selected")
                st.write(upload_data)
                st.subheader("Data after preprocessing")

                # clean text
                text_cleaning = lambda x: text_preprocessing(x)
                upload_data['Cleaned_Text'] = pd.DataFrame(upload_data[column[column.index(column_select)]].apply(text_cleaning))
                st.write(upload_data)

                def show_result(upload_data):
                    st.subheader("Result")
                    upload_data.loc[upload_data['Predicted'] == 0, 'Predicted'] = "Marah"
                    upload_data.loc[upload_data['Predicted'] == 1, 'Predicted'] = "Takut"
                    upload_data.loc[upload_data['Predicted'] == 2, 'Predicted'] = "Gembira"
                    upload_data.loc[upload_data['Predicted'] == 3, 'Predicted'] = "Cinta"
                    upload_data.loc[upload_data['Predicted'] == 4, 'Predicted'] = "Sedih"
                    upload_data.loc[upload_data['Predicted'] == 5, 'Predicted'] = "Terkejut"
                    st.write(upload_data)
                    st.markdown(download_csv(upload_data), unsafe_allow_html=True)

                # Predict
                if model_option == 'Xgboost':
                    pred_data=joblib_XGB_model.predict(upload_data['Cleaned_Text'])
                    upload_data['Predicted'] = pred_data
                    show_result(upload_data)
                elif model_option == 'Logistic Regression':
                    pred_data=joblib_LR_model.predict(upload_data['Cleaned_Text'])
                    upload_data['Predicted'] = pred_data
                    show_result(upload_data)
                elif model_option == 'Support Vector Classification':
                    pred_data=joblib_SVM_model.predict(upload_data['Cleaned_Text'])
                    upload_data['Predicted'] = pred_data
                    show_result(upload_data)
                elif model_option == 'Stochastic Gradient Descent':
                    pred_data=joblib_SGD_model.predict(upload_data['Cleaned_Text'])
                    upload_data['Predicted'] = pred_data
                    show_result(upload_data)

        else:
            st.subheader("Data after preprocessing")

            # clean text
            text_cleaning = lambda x: text_preprocessing(x)
            upload_data['Cleaned_Text'] = pd.DataFrame(upload_data[column[0]].apply(text_cleaning))
            st.write(upload_data)

            def show_result(upload_data):
                st.subheader("Result")
                upload_data.loc[upload_data['Predicted'] == 0, 'Predicted'] = "Marah"
                upload_data.loc[upload_data['Predicted'] == 1, 'Predicted'] = "Takut"
                upload_data.loc[upload_data['Predicted'] == 2, 'Predicted'] = "Gembira"
                upload_data.loc[upload_data['Predicted'] == 3, 'Predicted'] = "Cinta"
                upload_data.loc[upload_data['Predicted'] == 4, 'Predicted'] = "Sedih"
                upload_data.loc[upload_data['Predicted'] == 5, 'Predicted'] = "Terkejut"
                st.write(upload_data)
                st.markdown(download_csv(upload_data), unsafe_allow_html=True)

            # Predict
            if model_option == 'Xgboost':
                pred_data=joblib_XGB_model.predict(upload_data['Cleaned_Text'])
                upload_data['Predicted'] = pred_data
                show_result(upload_data)
            elif model_option == 'Logistic Regression':
                pred_data=joblib_LR_model.predict(upload_data['Cleaned_Text'])
                upload_data['Predicted'] = pred_data
                show_result(upload_data)
            elif model_option == 'Support Vector Classification':
                pred_data=joblib_SVM_model.predict(upload_data['Cleaned_Text'])
                upload_data['Predicted'] = pred_data
                show_result(upload_data)
            elif model_option == 'Stochastic Gradient Descent':
                pred_data=joblib_SGD_model.predict(upload_data['Cleaned_Text'])
                upload_data['Predicted'] = pred_data
                show_result(upload_data)
            
    else:
        st.warning("*Note:* One file must be uploaded!")