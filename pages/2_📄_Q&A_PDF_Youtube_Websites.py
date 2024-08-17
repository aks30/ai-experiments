import os
import aiutils
import requests
import traceback
import streamlit as st
import pickle
import validators
from streaming import StreamHandler
import re

from bs4 import BeautifulSoup
import io
import requests
from urllib.parse import urljoin, urlparse

from langchain.memory import ConversationBufferMemory
from langchain_core.documents.base import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (
    OpenAIWhisperParser
)


# Dictionary to store usernames and passwords
credentials = {
    "akash": "akashfortl",
    "kiran": "kiranfortl",
    "shyam": "shyamfortl",
    "khushboo": "khushbooforsl",
    "vijayashree": "vijayashreeforsl",
    "prateek": "prateekforsl",
    "tani": "tani"
}

st.set_page_config(page_title="Chat with Perosnal Knowledge Repository", page_icon="📄")
st.header('Chat with your PDFs, Wesbites/Articles and with Youtube videos')
st.write('This application stores each individuals resources in their specific scope which they can interact with. Note - Works for resources in English only.')

class CustomDataChatbot:

    def __init__(self):
        aiutils.sync_st_session()
        self.llm = aiutils.configure_llm()

    @staticmethod
    def login(username, password):
        if username in credentials and credentials[username] == password:
            return True
        return False
    
    @staticmethod
    @st.spinner('Saving to DB..')
    def save_emebeddings(self, splits):

        file_path = st.session_state.logged_in_user+"_"+"faiss_store_openai.pkl"

        # Create embeddings and store in vectordb
        embeddings = ""
        if(self.llm.model_name == 'llama3.1:8b'):
            embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        else:
            embeddings = OpenAIEmbeddings()

        # Check if the FAISS index file exists
        if os.path.exists(file_path):
            # Load existing FAISS index
            with open(file_path, "rb") as f:
                existing_bytes = pickle.load(f)
                existing_vectordb = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=existing_bytes,allow_dangerous_deserialization='true')
            
            # Add new documents to the existing index
            existing_vectordb.add_documents(splits)
            updated_vectordb = existing_vectordb
        else:
            # Create a new FAISS index if the file does not exist
            updated_vectordb = FAISS.from_documents(splits, embeddings)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(updated_vectordb.serialize_to_bytes(), f)
            return True
        
    @staticmethod
    @st.spinner('Reading from DB..')
    def retrieve_vectorstore(self):

        file_path = st.session_state.logged_in_user+"_"+"faiss_store_openai.pkl"

        if os.path.exists(file_path):
            # Define embeddings based on LLM
            embeddings = ""
            if(self.llm.model_name == 'llama3.1:8b'):
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            else:
                embeddings = OpenAIEmbeddings()

            with open(file_path, "rb") as f:
                vectorIndexSerialized = pickle.load(f)
                vectorstore = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=vectorIndexSerialized,allow_dangerous_deserialization='true')
                return vectorstore
        else:
            return False
            
    @staticmethod
    @st.spinner('Analysing documents..')
    def index_documents_to_vector_db(self,uploaded_files, pdfURL=""):
        
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Load documents
        docs = []
        if len(uploaded_files) > 0 :
            for file in uploaded_files:
                file_path = f'./{folder}/{file.name}'
                with open(file_path, 'wb') as f:
                    f.write(file.getvalue())
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
        
        if len(pdfURL.strip()) > 0:
            response = requests.get(pdfURL)
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the URL to extract the path
                parsed_url = urlparse(pdfURL)

                # Get the file name from the path
                file_name = os.path.basename(parsed_url.path)
                file_path = f'./{folder}/{file_name}'
                # Open a file in binary write mode at the specified path
                with open(file_path, 'wb') as file:
                    # Write the content of the response to the file
                    file.write(response.content)
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=['\n\n', '\n', '.', ',']
        )
        splits = text_splitter.split_documents(docs)

        vectorIndexProcess = self.save_emebeddings(self, splits)

        if(vectorIndexProcess == True):
            return True
        else:
            return False 


    @staticmethod
    def get_page_urls(url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
# Extract all <a> tags with href attributes
        links = [link.get('href') for link in soup.find_all('a', href=True)]
        
        # Normalize the base URL to handle variations like "www" and non-"www"
        parsed_url = urlparse(url)
        base_netloc = parsed_url.netloc.replace("www.", "")
        
        # Resolve relative URLs to absolute URLs and filter links within the same domain
        full_links = set()
        for link in links:
            # Resolve relative URLs
            full_url = urljoin(url, link)
            
            # Parse each URL to compare its netloc with the base domain
            parsed_link_url = urlparse(full_url)
            link_netloc = parsed_link_url.netloc.replace("www.", "")
            
            # Add the link if it belongs to the same domain
            if link_netloc == base_netloc:
                full_links.add(full_url)
        
        # Add the original URL to the set of links
        full_links.add(url)
        
        return full_links

    @staticmethod
    def scrape_website(url):
        content = ""
        try:
            base_url = "https://r.jina.ai/"
            final_url = base_url + url
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0'
                }
            response = requests.get(final_url, headers=headers)
            # Check if the request was successful
            if response.status_code == 200:
                content = response.text
        except Exception as e:
            traceback.print_exc()
        return content
    
    @staticmethod
    @st.spinner('Analysing URLs..')
    def index_URLs_to_vector_db(self,websites):
        
        # Regular expression to match YouTube video URLs
        youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
        # Compile the regex
        youtube_pattern = re.compile(youtube_regex)

        vectorIndexProcess = False
        # Scrape and load documents
        docs = []
        for mainURL in websites:
            if mainURL.endswith('.pdf'):
                uploadFileOperation = self.index_documents_to_vector_db(self,[],mainURL)
                if uploadFileOperation is True:
                    vectorIndexProcess = True
            elif (bool(youtube_pattern.match(mainURL)) == True) :
                processURLOperation = self.index_youtube_videos_to_vector_db(self,[mainURL])
                if processURLOperation is True:
                    vectorIndexProcess = True
            else:
                allSubURLs = self.get_page_urls(mainURL)
                for url in allSubURLs:
                    if url.endswith('.pdf'):
                        self.index_documents_to_vector_db(self,[],url)
                    else:
                        docs.append(Document(
                            page_content=self.scrape_website(url),
                            metadata={"source":url}
                            )
                        )

        if len(docs) > 0:   
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                separators=['\n\n', '\n', '.', ',']
            )
            splits = text_splitter.split_documents(docs)

            vectorIndexProcess = self.save_emebeddings(self, splits)

        if(vectorIndexProcess == True):
            return True
        else:
            return False 


    @staticmethod
    @st.spinner('Analysing Youtube Videos..')
    def index_youtube_videos_to_vector_db(self,urls):
        
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Scrape and load documents
        docs = []
        for url in urls:
            # Transcribe the videos to text
            loader = GenericLoader(
                YoutubeAudioLoader([url], folder), OpenAIWhisperParser()
            )
            transcripts = loader.load()
            docs.append(Document(
                page_content=transcripts[0].page_content,
                metadata={"source":url}
                )
            )
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=['\n\n', '\n', '.', ',']
        )
        splits = text_splitter.split_documents(docs)

        vectorIndexProcess = self.save_emebeddings(self, splits)

        if(vectorIndexProcess == True):
            return True
        else:
            return False 

    @staticmethod
    def setup_qa_chain(self):

        vectordb = self.retrieve_vectorstore(self)
        
        if vectordb is False:
            return False
        
        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':4, 'fetch_k':6}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        return qa_chain

    @aiutils.enable_chat_history
    def main(self):

        # Session state to keep track of login status
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False

        if not st.session_state.logged_in:
            st.sidebar.title("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                if self.login(username,password):
                    st.session_state.logged_in = True
                    st.session_state.logged_in_user = username
                    st.rerun()  # This will rerun the script and clear the sidebar
                else:
                    st.sidebar.error("Incorrect password")
        else:
            st.sidebar.write("You are logged in")

        
        if st.session_state.logged_in == True:
            # User Inputs
            uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
            if uploaded_files:
                uploadFileOperation = self.index_documents_to_vector_db(self,uploaded_files,"")
                if uploadFileOperation is True:
                    st.session_state["uploaded_files"] = []

            # User Inputs
            if "websites" not in st.session_state:
                st.session_state["websites"] = []
            

            web_url = st.sidebar.text_area(
                label='Enter Website URL(s) - Site Links, Article Links, Youtube Videos, PDF Links',
                placeholder="https://tunerlabs.com,https://www.youtube.com/watch?v=b4Dp_y-qsYg",
                help="Note, you can add multiple URLs which are comma separated."
                )
            

            if st.sidebar.button(":heavy_plus_sign: Process URL(s)"):

                # Split the string by commas
                urls = web_url.split(',')

                # Remove any leading/trailing whitespace from each part
                urls = [url.strip() for url in urls]

                for i, url in enumerate(urls, start=1):

                    valid_url = url.startswith('http') and validators.url(url)
                    if valid_url :
                        st.session_state["websites"].append(url)


            if len(st.session_state["websites"]) > 0 :
                processURLOperation = self.index_URLs_to_vector_db(self,st.session_state["websites"])
                if processURLOperation is True:
                    st.session_state["websites"] = []


            # User Inputs
            # if "youtube_video_links" not in st.session_state:
            #     st.session_state["youtube_video_links"] = []
            

            # youtube_video_urls = st.sidebar.text_area(
            #     label='Enter Youtube Video URL(s)',
            #     placeholder="https://www.youtube.com/watch?v=WY1jkYri2kUAD,https://www.youtube.com/watch?v=12WY1jkYri2kU",
            #     help="Note, you can add multiple URLs which are comma separated."
            #     )
            

            # if st.sidebar.button(":heavy_plus_sign: Process Youtube Video (s)"):

            #     # Split the string by commas
            #     urls = youtube_video_urls.split(',')

            #     # Remove any leading/trailing whitespace from each part
            #     urls = [url.strip() for url in urls]

            #     for i, url in enumerate(urls, start=1):

            #         valid_url = url.startswith('https://www.youtube.com/') and validators.url(url)
            #         if valid_url :
            #             st.session_state["youtube_video_links"].append(url)


            # if len(st.session_state["youtube_video_links"]) > 0 :
            #     processURLOperation = self.index_youtube_videos_to_vector_db(self,st.session_state["youtube_video_links"])
            #     if processURLOperation is True:
            #         st.session_state["youtube_video_links"] = []

            user_query = st.chat_input(placeholder="Ask me anything!")

            if user_query:
                qa_chain = self.setup_qa_chain(self)

                if qa_chain is False:
                    st.error("No data found in DB!")
                    st.stop()

                aiutils.display_msg(user_query, 'user')

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    result = qa_chain.invoke(
                        {"question":user_query},
                        {"callbacks": [st_cb]}
                    )
                    response = result["answer"]
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # to show references
                    for idx, doc in enumerate(result['source_documents'],1):
                        filename = os.path.basename(doc.metadata['source'])
                        if 'page' in doc.metadata:
                            page_num = doc.metadata['page']
                            ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                        else:
                            ref_title = f":blue[Reference {idx}: *{filename}*]"

                        with st.popover(ref_title):
                            st.caption(doc.page_content)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()