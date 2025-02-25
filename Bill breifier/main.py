import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Search for relevant content in the PDF
def search_pdf_by_tfidf(pdf_text, user_query):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sentences = pdf_text.split('\n')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences + [user_query])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = similarities.argmax()
    return sentences[best_match_idx]

# Summarize the content
def summarize_content(content):
    summarizer = pipeline("summarization", model="t5-small")
    
    summary = summarizer(content, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Generate MCQs (simplified)
def generate_mcqs(summary):
    questions = [
        {"question": "What is the main focus of the policy?", "options": ["Climate change", "Economic growth", "Healthcare", "Education"], "answer": "Climate change"},
        {"question": "Which year was the policy implemented?", "options": ["2010", "2015", "2020", "2025"], "answer": "2020"}
    ]
    return questions

# Complete flow for processing a scenario
def process_scenario(pdf_path, user_scenario):
    pdf_text = extract_text_from_pdf(pdf_path)
    relevant_content = search_pdf_by_tfidf(pdf_text, user_scenario)
    summarized_content = summarize_content(relevant_content)
    mcqs = generate_mcqs(summarized_content)
    return summarized_content, mcqs

# Example usage
pdf_path = "pdf/SixthAmmendment.pdf"
user_scenario = "What are the key points about climate change policy?"
summary, mcqs = process_scenario(pdf_path, user_scenario)

print("Summary of Relevant  Content:")
print(summary)

print("\nGenerated MCQs:")
for mcq in mcqs:
    print(f"Q: {mcq['question']}")
    print(f"Options: {', '.join(mcq['options'])}")
