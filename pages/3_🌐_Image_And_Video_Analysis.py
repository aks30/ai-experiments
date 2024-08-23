import streamlit as st
import aiutils
import pandas as pd
import requests  
from PIL import Image
import requests
import validators

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

st.set_page_config(page_title="Image and Video Analysis", page_icon="🌐")
st.header('Get an analysis of file types such as - text, document, images, videos to get their analysis.')
st.write('Download a sample CSV here and upload your sheet for analysis below.')


def login(username, password):
    if username in credentials and credentials[username] == password:
        return True
    return False

def process_image(image_url, question):

    image_processing_output = {
        "answers" : [],
        "caption" : "No caption available",
        "verification" : True
    }

    valid_url = image_url.startswith('http') and validators.url(image_url)
    if valid_url :

        questions = question.split(",")
        # Analyze the image
        # image_processing_output["answers"] = aiutils.blip1_image_qa(image_url, questions)
        
        if any(entry == "yes" for entry in image_processing_output["answers"]):
            image_processing_output["verification"] = True
        else:
            image_processing_output["verification"] = False
        
        image_processing_output["caption"] = aiutils.generate_image_caption(image_url)
    
    return image_processing_output


def process_doc(url, question):

    doc_processing_output = {
        "answers" : [],
        "caption" : "No caption available",
        "verification" : True
    }

    valid_url = url.startswith('http') and validators.url(url)
    if valid_url :

        questions = question.split(",")
        # Analyze the image
        # doc_processing_output["answers"] = aiutils.blip1_image_qa(url, questions)
        
        if any(entry == "yes" for entry in doc_processing_output["answers"]):
            doc_processing_output["verification"] = True
        else:
            doc_processing_output["verification"] = False
        
        doc_processing_output["caption"] = aiutils.generate_image_caption(url)
    
    return doc_processing_output


def main():

    # Session state to keep track of login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login(username,password):
                st.session_state.logged_in = True
                st.session_state.logged_in_user = username
                st.rerun()  # This will rerun the script and clear the sidebar
            else:
                st.sidebar.error("Incorrect password")
    else:
        st.sidebar.write("You are logged in")


    if st.session_state.logged_in == True:
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Add a new column for task evidence captions
            df['task_evidence_caption'] = "NA"

            # Add a new column for q&a
            df['task_evidence_question_answers'] = "NA"

            # Add a new column for task evidence test status
            df['task_evidence_verification'] = "Pass"

            # Add a new column for project evidence captions
            df['project_evidence_caption'] = "NA"

            # Add a new column for q&a
            df['project_evidence_question_answers'] = "NA"

            # Add a new column for task evidence test status
            df['project_evidence_verification'] = "Pass"

            
            # Process each URL in the CSV
            for index, row in df.iterrows():
                url = row['Task Evidence']
                taskEvidenceQuestion = row['Task Evidence Question']
                projectEvidenceQuestion = row['Project Evidence Question']
                if url.endswith(('.png', '.jpg', '.jpeg')):
                    st.subheader(f"Task Image {index+1}")
                    st.image(url, caption=f"Image {index+1}", use_column_width=True)
                    image_processing_output = process_image(url, taskEvidenceQuestion)
                    df.at[index, 'task_evidence_caption'] = image_processing_output["caption"]
                    df.at[index, 'task_evidence_question_answers'] = ",".join(image_processing_output["answers"])
                    df.at[index, 'task_evidence_verification'] = image_processing_output["verification"]
                    # st.write(image_processing_output)
                    st.write(image_processing_output["caption"])
                elif url.endswith('.mp4'):
                    # st.subheader(f"Task Video {index+1}")
                    # Videos are not directly supported for analysis by the Azure Vision API for captions
                    # st.video(url)
                    df.at[index, 'task_evidence_caption'] = "Video analysis is not supported in this demo."
                    df.at[index, 'task_evidence_question_answers'] = "NA"
                    df.at[index, 'task_evidence_verification'] = "NA"
                elif url.endswith(('.doc', '.pdf', '.txt', '.docx')):
                    # st.subheader(f"Task Video {index+1}")
                    # Videos are not directly supported for analysis by the Azure Vision API for captions
                    # st.video(url)
                    doc_processing_output = process_doc(url, taskEvidenceQuestion)
                    df.at[index, 'task_evidence_caption'] = doc_processing_output["caption"]
                    df.at[index, 'task_evidence_question_answers'] = ",".join(doc_processing_output["answers"])
                    df.at[index, 'task_evidence_verification'] = doc_processing_output["verification"]
                else:
                    # st.subheader(f"Documents {index+1}")
                    # Videos are not directly supported for analysis by the Azure Vision API for captions
                    # st.video(url)
                    description = "File type not supported."
                    df.at[index, 'task_evidence_caption'] = description
                    df.at[index, 'task_evidence_question_answers'] = "NA"
                    df.at[index, 'task_evidence_verification'] = "NA"
            
            # Display the result as a table
            st.subheader("Results")
            st.write(df)

if __name__ == "__main__":
    main()