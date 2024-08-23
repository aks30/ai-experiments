import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess

## Load an image

url = "https://bmzbbujw9kal.compat.objectstorage.ap-mumbai-1.oraclecloud.com/odev-dev-diksha-manage-learn/survey/659e5487ebc05700088b3625/00bcb191-a353-47c4-bf34-83a5d0e196f7/8f171324-74cd-4ff0-a136-5817cd8957e3/1704877755360.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")


# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP VQA model finetuned on VQAv2

vqa_model, vqa_vis_processors, vqa_txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

image = vqa_vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# uncomment to use base model
# caption_model, caption_vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )

# use "eval" processors for inference
# question = "Does the image have children in the photo?"
# question = vqa_txt_processors["eval"](question)

# samples = {"image": image, "text_input": question}
# vqa_model.predict_answers(samples=samples, inference_method="generate")

batch_size = 3

# create a batch of samples, could be multiple images or copies of the same image
image_batch = image.repeat(batch_size, 1, 1, 1)

# create a batch of questions, make sure the number of questions matches the number of images
question_1 = vqa_txt_processors["eval"]("Are there children in the image?")
question_2 = vqa_txt_processors["eval"]("Are there any posters seen in the image?")
question_3 = vqa_txt_processors["eval"]("Are kids playing in the photo?")

question_batch = [question_1, question_2, question_3]

print(vqa_model.predict_answers(samples={"image": image_batch, "text_input": question_batch}, inference_method="generate"))

# we associate a model with its preprocessors to make it easier for inference.
caption_model, caption_vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)
# uncomment to use base model
# caption_model, caption_vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )

def blip_captioning_example(url):
    raw_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # Image captioning using blip model
    caption_image = caption_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print(caption_model.generate({"image": caption_image}))

    print(caption_model.generate({"image": caption_image}, use_nucleus_sampling=True, num_captions=3))

blip_captioning_example(url)