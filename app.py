import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("🧠 AI Resume Matcher")
st.write("Upload your resume and a job description to see how well they align.")

# Uploads
resume_file = st.file_uploader("📄 Upload Resume (PDF or TXT)", type=["pdf", "txt"])
job_file = st.file_uploader("📝 Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    else:
        return file.read().decode("utf-8")

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    # Match score
    tfidf = TfidfVectorizer().fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    st.subheader(f"✅ Match Score: {round(score * 100, 2)}%")

    if score > 70:
        st.success("🔥 Great match! You’re a strong candidate.")
    elif score > 40:
        st.warning("🟡 Decent match, but could be stronger.")
    else:
        st.error("🔴 Weak match. Add more relevant skills or keywords.")
