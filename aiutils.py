import os
import openai
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import whisper
import ast
from clarifai.client.model import Model as clarifiaModel
from clarifai.client.input import Inputs as clarifaiInputs
# import torch
# from PIL import Image
# import requests
# from lavis.models import load_model_and_preprocess


# setup device to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load BLIP VQA model finetuned on VQAv2

# blip1_vqa_model, blip1_vqa_vis_processors, blip1_vqa_txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

# # we associate a model with its preprocessors to make it easier for inference.
# blip1_caption_model, blip1_caption_vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="large_coco", is_eval=True, device=device
# )

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key and model name from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT")
# Get OpenAI API key and model name from environment variables
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
calrifaiModelInstance = clarifiaModel(url="https://clarifai.com/microsoft/florence/models/florence-2-large",pat=CLARIFAI_PAT)

whispermodel = whisper.load_model("base")

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            if current_page == 'ContextChatbot.main':
                st.session_state["messages"] = [{"role": "assistant", "content": "My name is Mohini and I'm here to collect your micro improvement report. I understand that you have done some remarkable work and you are here to share the details with me. Once you share these details, I will create a report for you. Can we get started?"}]
            else:
                st.session_state["messages"] = [{"role": "assistant", "content": "Hi, there! Please use the left navigation to upload docments, website and youtube links."}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_llm():
    available_llms = ["gpt-4o-mini","llama3.1:8b","use your openai api key"]
    # llm_opt = st.sidebar.radio(
    #     label="LLM",
    #     options=available_llms,
    #     key="SELECTED_LLM"
    #     )
    llm_opt = "gpt-4o-mini"
    if llm_opt == "llama3.1:8b":
        llm = ChatOllama(model="llama3.1", base_url=OLLAMA_ENDPOINT)
    elif llm_opt == "gpt-4o-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=OPENAI_API_KEY)
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(model_name=model, temperature=0, streaming=True, api_key=openai_api_key)
    return llm

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v


def blip1_image_qa(image_url="",questions=[]):
    answers = []
    if len(image_url) > 0 and len(questions) > 0:

        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        image = blip1_vqa_vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # uncomment to use base model
        # caption_model, caption_vis_processors, _ = load_model_and_preprocess(
        #     name="blip_caption", model_type="base_coco", is_eval=True, device=device
        # )

        # use "eval" processors for inference
        # question = "Does the image have children in the photo?"
        # question = vqa_txt_processors["eval"](question)

        # samples = {"image": image, "text_input": question}
        # vqa_model.predict_answers(samples=samples, inference_method="generate")

        batch_size = len(questions)

        # create a batch of samples, could be multiple images or copies of the same image
        image_batch = image.repeat(batch_size, 1, 1, 1)

        # create a batch of questions, make sure the number of questions matches the number of images
        question_batch = []
        for question in questions:
            question_batch.append(blip1_vqa_txt_processors["eval"](question))

        answers = blip1_vqa_model.predict_answers(samples={"image": image_batch, "text_input": question_batch}, inference_method="generate")


    return answers


def blip1_image_captioning(image_url=""):
    caption = ""
    if len(image_url) > 0:

        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        # Image captioning using blip model
        caption_image = blip1_caption_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = blip1_caption_model.generate({"image": caption_image})

        # print(caption_model.generate({"image": caption_image}, use_nucleus_sampling=True, num_captions=3))

    return caption



def generate_image_caption(image_url=""):
    caption = ""
    if len(image_url) > 0:
        prompt = "<MORE_DETAILED_CAPTION>"
        inference_params = dict(max_tokens=512)
            
        caption_prediction = calrifaiModelInstance.predict(inputs = [clarifaiInputs.get_multimodal_input(input_id="",image_url=image_url, raw_text=prompt)],inference_params=inference_params)
            
        caption_output = caption_prediction.outputs[0].data.text.raw

        # Convert the string representation of a dictionary into an actual dictionary
        caption_output_dict = ast.literal_eval(caption_output)

        # Now, you can access the value using the key
        caption = caption_output_dict['<MORE_DETAILED_CAPTION>']

    return caption


def speech_to_text(audio_file):
    transcript = ""
    # result = whispermodel.transcribe(audio_file)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whispermodel.device)

    # detect the spoken language
    _, probs = whispermodel.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(task="translate")
    result = whisper.decode(whispermodel, mel, options)

    transcript = result.text
    return transcript