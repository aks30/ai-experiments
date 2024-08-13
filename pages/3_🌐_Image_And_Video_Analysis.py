import streamlit as st
import pandas as pd
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration  
from PIL import Image
import requests
import validators
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

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

def process_image(image_url, question):

    valid_url = image_url.startswith('http') and validators.url(image_url)
    if valid_url :
        # Analyze the image
        description_results = analyse_image(image_url, question)
        
        if description_results:
            return description_results
        else:
            return "No description available."
    else:
        return "No description available."

def analyse_image(url, question):

    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    inputs = processor(image, question, return_tensors="pt")

    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True).strip()

    return answer

def main():

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Add a new column for captions
        df['caption'] = ""
        
        # Process each URL in the CSV
        for index, row in df.iterrows():
            url = row['Task Evidence']
            taskEvidenceQuestion = row['Task Evidence Question']
            projectEvidenceQuestion = row['Project Evidence Question']
            if url.endswith(('.png', '.jpg', '.jpeg')):
                st.subheader(f"Image {index+1}")
                st.image(url, caption=f"Image {index+1}", use_column_width=True)
                description = process_image(url, taskEvidenceQuestion)
                df.at[index, 'caption'] = description
                st.write(description)
            elif url.endswith('.mp4'):
                st.subheader(f"Video {index+1}")
                # Videos are not directly supported for analysis by the Azure Vision API for captions
                st.video(url)
                df.at[index, 'caption'] = "Video analysis is not supported in this demo."
            else:
                st.subheader(f"Documents {index+1}")
                # Videos are not directly supported for analysis by the Azure Vision API for captions
                # st.video(url)
                description = "Analysing documents."
                df.at[index, 'caption'] = description
        # Display the result as a table
        st.subheader("Results")
        st.write(df)

if __name__ == "__main__":
    main()