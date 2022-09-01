import streamlit as st
import pickle

tfidf= pickle.load(open('vectorizer_pkl','rb'))
text_transform= pickle.load(open('model_pkl','rb'))

st.title('SMS/EMAIL SPAM DETECTOR')


st.write('The spam detection is a big issue in mobile message communication due to which mobile message communication is insecure. In order to tackle this problem, an accurate and precise method is needed to detect the spam in mobile message communication. We proposed the applications of the machine learning-based spam detection method for accurate detection.')
st.write('Built with streamlit and Python ')
input_sms=st.text_input('Enter your text here')
if st.button("predict"):
    st.text("predict  {}\n".format(input_sms))
    cv_text= tfidf.transform([input_sms]).toarray()
    prediction= text_transform.predict(cv_text)
    if prediction== 0:
        st.write("Not spam")
    else:
        st.write("spam")









