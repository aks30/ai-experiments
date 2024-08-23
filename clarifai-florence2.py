import os
import ast

from clarifai.client.model import Model
from clarifai.client.input import Inputs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key and model name from environment variables
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

prompt = "<MORE_DETAILED_CAPTION>"
image_url = "https://bmzbbujw9kal.compat.objectstorage.ap-mumbai-1.oraclecloud.com/odev-dev-diksha-manage-learn/survey/659e5487ebc05700088b3625/00bcb191-a353-47c4-bf34-83a5d0e196f7/8f171324-74cd-4ff0-a136-5817cd8957e3/1704877755360.jpg"
inference_params = dict(max_tokens=512)
    
model_prediction = Model(url="https://clarifai.com/microsoft/florence/models/florence-2-large",pat=CLARIFAI_PAT).predict(inputs = [Inputs.get_multimodal_input(input_id="",image_url=image_url, raw_text=prompt)],inference_params=inference_params)
    
model_output = model_prediction.outputs[0].data.text.raw

print(type(model_output))

# Convert the string representation of a dictionary into an actual dictionary
model_output_dict = ast.literal_eval(model_output)

# Now, you can access the value using the key
caption = model_output_dict['<MORE_DETAILED_CAPTION>']
print(caption)


