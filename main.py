import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# nlp
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

import difflib
import random

import gradio as gr
from difflib import SequenceMatcher
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import time
import os
import base64
import random

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_excel('icthub_dataset.xlsx')

# Get random questions for suggestions
def get_suggested_questions(n=3):
    # Filter out greeting questions
    non_greeting = df[df['Category'] != 'Greeting']
    questions = non_greeting['User Input'].unique().tolist()
    return random.sample(questions, min(n, len(questions)))

# Preprocessing function
def preprocessing(text):
    text = text.lower()
    text = word_tokenize(text)
    filtered_text = [word for word in text if word.isalnum()]
    filtered_text = [word for word in filtered_text if word not in stopwords.words('english')]
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in filtered_text]
    return " ".join(stemmed)

# Prepare chatbot knowledge base
df['Preprocessed'] = df['User Input'].apply(preprocessing)
chatbot_knowledge = dict(zip(df['Preprocessed'], df['Chatbot Response']))

# Response generation
def get_response(user_input, threshold=0.6):
    processed_input = preprocessing(user_input)
    best_match = None
    highest_similarity = 0
    for question in chatbot_knowledge:
        similarity = SequenceMatcher(None, processed_input, question).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = question
    if highest_similarity >= threshold:
        return chatbot_knowledge[best_match]
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

# Chat response function
def respond(message, chat_history):
    bot_message = get_response(message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history

# Function to handle suggestion clicks
def suggestion_click(question, chat_history):
    # Add user question to chat
    chat_history.append((question, ""))
    # Get bot response
    bot_message = get_response(question)
    chat_history[-1] = (question, bot_message)
    return chat_history, gr.update(visible=False)

# Function to start new conversation
def new_conversation():
    return [
        [(None, "Hi! I'm ICTHub Assistant. How can I help you today?")],  # Reset chat
        gr.update(visible=True),  # Show suggestions
        ""  # Clear message box
    ]

# GUI with enhanced logo header
with gr.Blocks(theme=gr.themes.Default(primary_hue="slate", neutral_hue="slate")) as demo:
    # Logo header with left-aligned image
    gr.HTML("""
    <div style="
        padding: 25px 20px;
        border-bottom: 1px solid #333;
    ">
        <img src="data:image/png;base64,{}"
             alt="ICTHub Logo"
             style="
                height: 90px;
                max-width: 100%;
                object-fit: contain;
                display: block;
                margin: 0;
             ">
    </div>
    """.format(base64.b64encode(open("icthub_logo.png", "rb").read()).decode()))

    with gr.Column(elem_classes="main-container"):
        chatbot = gr.Chatbot(
            value=[(None, "Hi! I'm ICTHub Assistant. How can I help you today?")],
            bubble_full_width=False,
            height="calc(100vh - 220px)",
            avatar_images=(None, "openAI_logo.jpg"),
            label=None
        )

        # Suggested questions component
        with gr.Column(visible=True, elem_classes="suggestions-container") as suggestions:
            gr.Markdown("### What would you like to know about ICTHub?", elem_classes="suggestions-title")
            with gr.Row():
                suggested_questions = get_suggested_questions(3)
                for question in suggested_questions:
                    btn = gr.Button(
                        question,
                        elem_classes="suggestion-btn",
                        size="sm",
                    )
                    # Set up click handler for each button
                    btn.click(
                        fn=lambda q=question: suggestion_click(q, chatbot.value),
                        outputs=[chatbot, suggestions]
                    )

        with gr.Row(elem_classes="input-area"):
            msg = gr.Textbox(
                placeholder="Type your message...",
                show_label=False,
                container=False,
                scale=8,
                elem_classes="textbox"
            )
            # New Conversation button instead of clear button
            new_chat_btn = gr.Button(
                "New Conversation",
                elem_classes="new-chat-btn",
                scale=2
            )

        msg.submit(
            respond,
            [msg, chatbot],
            [msg, chatbot]
        ).then(
            lambda chat_history: gr.update(visible=len(chat_history) <= 1),
            [chatbot],
            [suggestions]
        )

        # New conversation button click handler
        new_chat_btn.click(
            new_conversation,
            outputs=[chatbot, suggestions, msg]
        )

    demo.css = """
    :root {
        --background-fill-primary: #0a0a0a;
        --background-fill-secondary: #1a1a1a;
        --color-background-primary: #0a0a0a;
    }

    body, html {
        margin: 0;
        padding: 0;
        height: 100%;
    }

    .gradio-container {
        background-color: #0a0a0a !important;
        font-family: 'Segoe UI', sans-serif;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh;
    }

    .main-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 140px) !important;
        padding: 0 !important;
    }

    .gr-chatbot {
        background-color: #1a1a1a !important;
        border-radius: 0 !important;
        padding: 20px;
        width: 100% !important;
        margin: 0 !important;
        flex-grow: 1;
        overflow-y: auto !important;
    }

    .input-area {
        background-color: #1a1a1a !important;
        padding: 16px 20px !important;
        border-top: 1px solid #333 !important;
        width: 100% !important;
        margin: 0 !important;
        position: sticky;
        bottom: 0;
    }

    .textbox textarea {
        background-color: #2a2a2a !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 18px !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        min-height: 20px !important;
    }

    .new-chat-btn {
        background-color: #333 !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 12px 20px !important;
        height: 100% !important;
    }

    .new-chat-btn:hover {
        background-color: #444 !important;
    }

    .message {
        padding: 12px 16px;
        margin: 8px 0;
    }

    .message.user {
        background-color: #2a2a2a !important;
        color: white !important;
        border-radius: 18px 18px 0 18px !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }

    .message.bot {
        background-color: #333333 !important;
        color: white !important;
        border-radius: 18px 18px 18px 0 !important;
        margin-right: auto !important;
        max-width: 80% !important;
    }

    .avatar img {
        width: 32px !important;
        height: 32px !important;
        object-fit: contain !important;
    }

    footer {
        display: none !important;
    }

    .gradio-container .gr-html {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Suggestions styling */
    .suggestions-container {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(26, 26, 26, 0.9);
        padding: 20px;
        border-radius: 12px;
        width: 80%;
        max-width: 600px;
        text-align: center;
        z-index: 100;
    }

    .suggestions-title {
        color: white !important;
        margin-bottom: 20px !important;
    }

    .suggestion-btn {
        background-color: #333 !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 12px 16px !important;
        margin: 0 8px !important;
        flex-grow: 1;
        white-space: normal !important;
        height: auto !important;
        min-height: 50px !important;
    }

    .suggestion-btn:hover {
        background-color: #444 !important;
    }
    """

if __name__ == "__main__":
    demo.launch(share=True)
