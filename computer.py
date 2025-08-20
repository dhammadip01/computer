import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import base64

# ---------- Spam Dataset ----------
data = pd.read_csv("C:\\Users\\Dhammadip\\Downloads\\spam.csv")
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

# ---------- Embed Beep Sound (base64) ----------
BEEP_BASE64 = """
UklGRjQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAAAgP8AgP8AgP8AgP8A
gP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8A
gP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8A
gP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8A
"""  # Short beep wav

def play_beep():
    # Convert base64 → src for HTML audio
    beep_bytes = base64.b64decode(BEEP_BASE64)
    b64sound = base64.b64encode(beep_bytes).decode()
    audio_tag = f"""
        <audio autoplay loop>
        <source src="data:audio/wav;base64,{b64sound}" type="audio/wav">
        </audio>
        <script>
        setTimeout(() => {{
            document.querySelectorAll('audio').forEach(a => a.remove());
        }}, 5000);  // stop after 5 seconds
        </script>
    """
    st.markdown(audio_tag, unsafe_allow_html=True)

# ---------- Streamlit UI ----------
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
                play_beep()  # 🔔 Auto beep in browser for 5 sec
            else:
                st.markdown(f"### ✅ Result: **{output}**", unsafe_allow_html=True)
