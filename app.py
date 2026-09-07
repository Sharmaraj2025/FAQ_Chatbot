import string
import nltk
import streamlit as st
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Fast NLTK Setup
@st.cache_resource
def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

init_nltk()

# Gemini API Setup
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# Local FAQ Dataset
faqs = [
    {
        "question": "What is this chatbot for?",
        "answer": "I am an AI-powered FAQ chatbot designed to answer your common questions instantly!"
    },
    {
        "question": "What are your operational hours?",
        "answer": "Our services are available 24/7 online. Customer support operates from 9 AM to 6 PM."
    },
    {
        "question": "How can I reset my password?",
        "answer": "Click on 'Forgot Password' on the login page and follow the instructions sent to your email."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach customer support via email at support@example.com or call our toll-free number."
    },
    {
        "question": "What services do you offer?",
        "answer": "We offer AI integration, software development, web application design, and technical training."
    }
]

questions = [item['question'] for item in faqs]
answers = [item['answer'] for item in faqs]

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# 2. Cached Vectors (Fast Matching)
@st.cache_data
def get_faq_vectors():
    preprocessed_questions = [preprocess_text(q) for q in questions]
    vectorizer = TfidfVectorizer()
    faq_matrix = vectorizer.fit_transform(preprocessed_questions)
    return vectorizer, faq_matrix

vectorizer, faq_matrix = get_faq_vectors()

# 3. Stream Generator with Auto-Model Selection
def stream_gemini_response(prompt):
    models_to_try = ['gemini-3.6-flash', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
    success = False
    last_error = ""

    for model_name in models_to_try:
        try:
            ai_model = genai.GenerativeModel(model_name)
            response = ai_model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            success = True
            break
        except Exception as e:
            last_error = str(e)
            continue

    if not success:
        yield f"Unable to reach AI right now: {last_error}"

# Hybrid Matching Handler
def get_response_stream(user_query):
    cleaned_query = preprocess_text(user_query)
    if not cleaned_query.strip():
        yield "Please ask a valid question."
        return

    query_vec = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vec, faq_matrix).flatten()
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]

    if best_score > 0.3:
        yield f"**(FAQ Match)** {answers[best_match_idx]}"
    else:
        yield from stream_gemini_response(user_query)

# Streamlit Chat UI Setup
st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖")
st.title("🤖 Universal AI & FAQ Chatbot")
st.caption("Lightning fast answers using TF-IDF Caching & Streaming Output.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response_text = st.write_stream(get_response_stream(user_input))

    st.session_state.messages.append({"role": "assistant", "content": response_text})