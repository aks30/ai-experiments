import streamlit as st
import pandas as pd
import requests  
from PIL import Image
import requests
import validators
import torch

# Load model directly
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

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

    answer = ''
    raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    # use "eval" processors for inference
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)

    samples = {"image": image, "text_input": question}

    answer = model.predict_answers(samples=samples, inference_method="generate")
    # if len(question) > 0:
    #     prompt = "Question: "+question+" Answer:"
        
    #     inputs = processor(images=image, text=prompt, return_tensors="pt")

    #     generated_ids = model.generate(**inputs)
    #     answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    # else:

    #     inputs = processor(images=image, return_tensors="pt")
    #     generated_ids = model.generate(**inputs)
    #     answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    
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