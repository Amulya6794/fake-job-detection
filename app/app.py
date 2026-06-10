import streamlit as st
import pickle

# Load trained model and tfidf
model = pickle.load(open("../model/model.pkl", "rb"))
tfidf = pickle.load(open("../model/tfidf.pkl", "rb"))
st.title("🧠 Fake Job Detection System")

job_text = st.text_area("Enter Job Description")

if st.button("Predict"):

    if job_text.strip() == "":
        st.warning("Please enter a job description")

    else:
        # Transform input
        vector = tfidf.transform([job_text])

        # Predict
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)

        # Debug info (VERY IMPORTANT)
        st.write("Raw Prediction:", prediction)
        st.write("Confidence:", proba)

        # Result logic (IMPORTANT FIX)
        if prediction == 1:
            st.error("❌ Fake Job Posting Detected")
        else:
            st.success("✅ Genuine Job Posting")