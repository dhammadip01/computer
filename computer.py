import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import winsound  # 🔔 For beep sound

data = pd.read_csv("spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']

mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)
model = MultinomialNB()
model.fit(features, cat_train)


def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

st.markdown("<h1 style='color: red; text-align: center;'>PRESENTED BY DHAMMADIP</h1>", unsafe_allow_html=True)


USERNAME = "dhammadip"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("✅ Login successful! Now you can use the Spam Detector.")
        else:
            st.error("❌ Invalid username or password")
else:
    
    st.header('📩 Spam Detection') 

    input_mess = st.text_input('Enter Message Here')

    if st.button('Validate'):
        if input_mess.strip() == "":
            st.warning("⚠ Please enter a message to check.")
        else:
            output = predict(input_mess)
            if output == "Spam":
                st.markdown(f"### 🚨 Result: **{output}**", unsafe_allow_html=True)
                # 🔔 Beep sound for 5 seconds (1000 Hz frequency)
                winsound.Beep(1000, 5000)
            else:
                st.markdown(f"### ✅ Result: **{output}**", unsafe_allow_html=True)

