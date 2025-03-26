import os
import pathlib
import subprocess
import sys
import warnings
from typing import Dict, Any, List

# --- Critical Imports with Auto-Install ---
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    # Install before importing Streamlit to avoid spinner issues
    subprocess.run([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "--upgrade", 
        "transformers", 
        "torch",
        "--quiet"
    ], check=True)
    from transformers import AutoTokenizer, AutoModel

# Now safe to import Streamlit
import streamlit as st
from tokenizers.normalizers import BertNormalizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import plotly.graph_objects as go
import plotly.express as px

# Set page config to wide layout at the start
st.set_page_config(
    layout="wide",
    page_title="Alloy Based Chatbot",
    page_icon="🔍"
)

warnings.filterwarnings('ignore')

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = st.secrets["google"]["GOOGLE_API_KEY"]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'question' not in st.session_state:
    st.session_state.question = ''
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_context' not in st.session_state:
    st.session_state.selected_context = None

file_path = "vocab_mappings.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    mappings = f.read().strip().split('\n')

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

# Define the prompt template
template = """
You are an intelligent assistant designed to provide accurate and helpful answers based on the context provided. Follow these guidelines:
1. Use only the information from the context to answer the question.
2. If the context does not contain enough information to answer the question, say "I don't know" and do not make up an answer.
3. Be concise and specific in your response.
4. Always end your answer with "Thanks for asking!" to maintain a friendly tone.

Context: {context}

Question: {question}

Answer:
"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

class State:
    def __init__(self, question: str):
        self.question = question
        self.context: List[Document] = []
        self.answer: str = ""

def load_embeddings_from_csv(file_path: str):
    print(f"Loading embeddings from CSV file: {file_path}")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    print("Embeddings loaded successfully.")
    return df

def generate_query_embedding(query_text: str, model_name: str):
    print(f"Generating query embedding using {model_name}...")
    if model_name == "matscibert":
        return generate_matscibert_embedding(query_text)
    elif model_name == "bert":
        return generate_bert_embedding(query_text)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def generate_matscibert_embedding(query_text: str):
    print("Generating Matscibert embedding...")
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    model = AutoModel.from_pretrained('m3rg-iitd/matscibert')

    norm_sents = [normalize(query_text)]
    tokenized_sents = tokenizer(norm_sents, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        last_hidden_state = model(**tokenized_sents).last_hidden_state

    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
    print("Matscibert embedding generated.")
    return sentence_embedding

def generate_bert_embedding(query_text: str):
    print("Generating BERT embedding...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained("bert-base-uncased")

    encoded_input = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded_input)

    sentence_embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    print("BERT embedding generated.")
    return sentence_embedding

def retrieve(state: State, embeddings_df: pd.DataFrame, model_name: str):
    print("Retrieving relevant documents...")
    query_embedding = generate_query_embedding(state.question, model_name)
    document_embeddings = np.array(embeddings_df['embedding'].tolist())
    similarities = cosine_similarity([query_embedding], document_embeddings)
    top_indices = similarities.argsort()[0][::-1]
    state.context = [Document(page_content=embeddings_df.iloc[i]['document']) for i in top_indices[:3]]
    print("Documents retrieved.")
    return state

def generate(state: State):
    print("Generating answer based on context and question...")
    docs_content = "\n\n".join(doc.page_content for doc in state.context)
    messages = custom_rag_prompt.invoke({"question": state.question, "context": docs_content})
    response = model.invoke(messages)
    state.answer = response.content
    print("Answer generated.")
    return state

def workflow(state_input: Dict[str, Any], embeddings_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    print(f"Running workflow for question: {state_input['question']} with model: {model_name}")
    state = State(state_input["question"])
    state = retrieve(state, embeddings_df, model_name)
    state = generate(state)
    print(f"Workflow complete for question: {state_input['question']}.")
    return {"context": state.context, "answer": state.answer}

def compute_bertscore(answer: str, context: str) -> Dict[str, float]:
    P, R, F1 = bert_score([answer], [context], lang="en")
    return {
        "BERTScore Precision": P.mean().item(),
        "BERTScore Recall": R.mean().item(),
        "BERTScore F1": F1.mean().item()
    }

def compute_rouge(answer: str, context: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(context, answer)
    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure
    }

def evaluate_answer(answer: str, context: str) -> Dict[str, Dict[str, float]]:
    return {
        "BERTScore": compute_bertscore(answer, context),
        "ROUGE": compute_rouge(answer, context)
    }

@st.cache_resource
def load_data():
    matscibert_csv = 'matscibert_embeddings.csv'
    bert_csv = 'bert_embeddings.csv'
    embeddings_df_matscibert = load_embeddings_from_csv(matscibert_csv)
    embeddings_df_bert = load_embeddings_from_csv(bert_csv)
    return embeddings_df_matscibert, embeddings_df_bert

embeddings_df_matscibert, embeddings_df_bert = load_data()

def ask_question(question: str):
    print(f"Asking question: {question}")
    matscibert_result = workflow({"question": question}, embeddings_df_matscibert, model_name="matscibert")
    bert_result = workflow({"question": question}, embeddings_df_bert, model_name="bert")

    matscibert_context = "\n\n".join(doc.page_content for doc in matscibert_result["context"])
    matscibert_answer = matscibert_result["answer"]
    matscibert_scores = evaluate_answer(matscibert_answer, matscibert_context)

    bert_context = "\n\n".join(doc.page_content for doc in bert_result["context"])
    bert_answer = bert_result["answer"]
    bert_scores = evaluate_answer(bert_answer, bert_context)

    return {
        "matscibert": {
            "Context": matscibert_context,
            "Answer": matscibert_answer,
            "Scores": matscibert_scores
        },
        "bert": {
            "Context": bert_context,
            "Answer": bert_answer,
            "Scores": bert_scores
        }
    }

def create_bertscore_chart(scores: Dict[str, float]):
    metrics = ['Precision', 'Recall', 'F1']
    values = [scores['BERTScore Precision'], scores['BERTScore Recall'], scores['BERTScore F1']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=['#4285F4', '#34A853', '#FBBC05'],
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='BERTScore Metrics',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_rouge_chart(scores: Dict[str, float]):
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    values = [scores['ROUGE-1'], scores['ROUGE-2'], scores['ROUGE-L']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=['#EA4335', '#34A853', '#FBBC05'],
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='ROUGE Metrics',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_comparison_chart(matscibert_scores: Dict[str, Dict[str, float]], bert_scores: Dict[str, Dict[str, float]]):
    metrics = ['Precision', 'Recall', 'F1', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    matscibert_values = [
        matscibert_scores['BERTScore']['BERTScore Precision'],
        matscibert_scores['BERTScore']['BERTScore Recall'],
        matscibert_scores['BERTScore']['BERTScore F1'],
        matscibert_scores['ROUGE']['ROUGE-1'],
        matscibert_scores['ROUGE']['ROUGE-2'],
        matscibert_scores['ROUGE']['ROUGE-L']
    ]
    
    bert_values = [
        bert_scores['BERTScore']['BERTScore Precision'],
        bert_scores['BERTScore']['BERTScore Recall'],
        bert_scores['BERTScore']['BERTScore F1'],
        bert_scores['ROUGE']['ROUGE-1'],
        bert_scores['ROUGE']['ROUGE-2'],
        bert_scores['ROUGE']['ROUGE-L']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=matscibert_values,
        name='Matscibert',
        marker_color='#4285F4'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=bert_values,
        name='BERT',
        marker_color='#EA4335'
    ))
    
    fig.update_layout(
        title='Model Comparison',
        barmode='group',
        height=500
    )
    
    return fig

def home_page():
    # CSS to center content vertically from middle to bottom
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 70vh;
        }
        @media (max-height: 700px) {
            .main .block-container {
                min-height: 80vh;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Centered heading
    st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <h1>Welcome to the Alloy Based Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Search components - centered in the middle of available space
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user_input = st.text_area(
            "Enter your question about alloys:", 
            key="user_input", 
            value=st.session_state.question,
            height=100,
            label_visibility="collapsed",
            placeholder="Ask your question here"
        )
        
        submit_button = st.button(
            "Search", 
            key="search_button",
            use_container_width=True
        )
    
    if submit_button and user_input:
        st.session_state.question = user_input
        st.session_state.results = ask_question(user_input)
        st.session_state.page = 'results'
        st.rerun()

def results_page():
    st.title("Search Results")
    
    if st.session_state.results:
        results = st.session_state.results
        
        # First show answers in columns
        st.subheader("Model Answers")
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("### Matscibert Answer")
                st.write(results["matscibert"]["Answer"])
        
        with col2:
            with st.container(border=True):
                st.markdown("### BERT Answer")
                st.write(results["bert"]["Answer"])
        
        # Then show the comparison chart
        st.subheader("Model Performance Comparison")
        st.plotly_chart(
            create_comparison_chart(results["matscibert"]["Scores"], results["bert"]["Scores"]),
            use_container_width=True
        )
        
        # Detailed metrics in tabs
        st.subheader("Detailed Metrics")
        tab1, tab2 = st.tabs(["Matscibert Metrics", "BERT Metrics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_bertscore_chart(results["matscibert"]["Scores"]["BERTScore"]),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    create_rouge_chart(results["matscibert"]["Scores"]["ROUGE"]),
                    use_container_width=True
                )
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_bertscore_chart(results["bert"]["Scores"]["BERTScore"]),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    create_rouge_chart(results["bert"]["Scores"]["ROUGE"]),
                    use_container_width=True
                )
    
    # Navigation buttons at the bottom
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start New Search", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.question = ''
            st.rerun()
    with col2:
        if st.button("View Context", use_container_width=True):
            st.session_state.page = 'context_choice'
            st.rerun()

def context_choice_page():
    st.title("Select Context to View")
    
    st.write("Choose which model's context you'd like to examine:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Matscibert Context", use_container_width=True):
            st.session_state.selected_context = "matscibert"
            st.session_state.page = 'context_view'
            st.rerun()
    with col2:
        if st.button("View BERT Context", use_container_width=True):
            st.session_state.selected_context = "bert"
            st.session_state.page = 'context_view'
            st.rerun()
    
    st.markdown("---")
    if st.button("Back to Results", use_container_width=True):
        st.session_state.page = 'results'
        st.rerun()

def context_view_page():
    st.title(f"{st.session_state.selected_context.capitalize()} Context")
    
    # Context switching buttons at top
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Switch to Matscibert Context", 
                    disabled=st.session_state.selected_context == "matscibert",
                    use_container_width=True):
            st.session_state.selected_context = "matscibert"
            st.rerun()
    with col2:
        if st.button("Switch to BERT Context", 
                    disabled=st.session_state.selected_context == "bert",
                    use_container_width=True):
            st.session_state.selected_context = "bert"
            st.rerun()
    
    # Display the context in a scrollable container
    if st.session_state.results and st.session_state.selected_context:
        context = st.session_state.results[st.session_state.selected_context]["Context"]
        with st.container(height=600, border=True):
            st.markdown(f"```\n{context}\n```")
    
    # Navigation buttons at bottom
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    with col2:
        if st.button("New Search", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.question = ''
            st.rerun()

def main():
    # Add some custom CSS
    st.markdown("""
    <style>
    /* Search bar styling */
    .stTextArea textarea {
        min-height: 100px;
        border: none !important;
        box-shadow: none !important;
        padding: 12px !important;
    }
    .stTextArea div[data-baseweb="base-input"] {
        border-radius: 8px !important;
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        margin-top: 0.5rem;
    }
    
    /* Layout adjustments */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
    }
    
    /* Remove extra padding */
    .main .block-container {
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'results':
        results_page()
    elif st.session_state.page == 'context_choice':
        context_choice_page()
    elif st.session_state.page == 'context_view':
        context_view_page()

if __name__ == "__main__":
    main()
