import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon='💬',
    layout='wide'
)

st.header("AI powered Chatbot Implementations")

st.write("""
This application is leveraging OpenAI and Llama based on selection.

- **Mohini**: Engage in interview format to record your improvement journey. It summarises and shares a report of the project at the end.
- **Q&A_PDF_Youtube_Websites**: A chatbot that acts as your knowledge repository. This is specific to each individual. You can add PDFs, Websites and Youtube video links and ask questions about the same. This repository grows over time. Note - This is for resources in English only.

To explore sample usage of each chatbot, please navigate to the corresponding chatbot section.
         
Please provided your feedback to - aks301190@gmail.com
""")