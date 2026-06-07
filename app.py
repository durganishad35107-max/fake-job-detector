import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
 model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
 tfidf = pickle.load(f)

st.title("Fake Job Detector")
st.write("Paste a job description below to check if it is real or fake.")

job_text = st.text_area("Job Description", height=200)

if st.button("Analyze"):
 if job_text.strip() == "":
 st.warning("Please enter a job description!")
 else:
 vec = tfidf.transform([job_text])
 pred = model.predict(vec)[0]
 prob = model.predict_proba(vec)[0]
 if pred == 1:
 st.error(f"This looks like a FAKE job posting. Confidence: {round(prob[1]*100)}%")
 else:
 st.success(f"This looks like a REAL job posting. Confidence: {round(prob[0]*100)}%")
