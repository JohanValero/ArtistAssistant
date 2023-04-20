import os
import torch
import base64

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel

print("Loading model.")
cCLIP_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
cCLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
cCLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
cCLIP_MODEL = cCLIP_MODEL.to(cCLIP_DEVICE)
print(f"Model loaded in device {cCLIP_DEVICE}.")

app = Flask(__name__)

def generate_json_response(iJson):
    vResponse = jsonify(iJson)
    #vResponse.headers.add('Access-Control-Allow-Origin', '*')
    return vResponse

@app.route("/", methods = ["GET"])
def home():
    try:
        vJson = {"message": "¡Server funcionando correctamente!"}
        return generate_json_response(vJson)
    except Exception as ex:
        print(ex)
        raise ex

@app.route("/genera/clip", methods = ["POST"])
def detect_object():
    try:
        vRequestJson = request.get_json()
        vTags = vRequestJson['tags']
        vImageBase64 = vRequestJson['image']
        vImageBase64 = base64.b64decode(vImageBase64)
        vImage = Image.open(BytesIO(vImageBase64))

        vInputs = cCLIP_PROCESSOR(
            text = vTags,
            images = vImage,
            return_tensors = "pt",
            padding = True
        )
        vInputs.to(cCLIP_DEVICE)

        vOutPuts = cCLIP_MODEL(** vInputs)
        vLogits_per_image = vOutPuts.logits_per_image       # this is the image-text similarity score
        vProbabilitys = vLogits_per_image.softmax(dim = 1)  # we can take the softmax to get the label probabilities

        vJsonResponse = {
            'result': 'success',
            'data': vProbabilitys.tolist()[0]
        }
        return generate_json_response(vJsonResponse)
    except Exception as ex:
        print(ex)
        raise ex

if __name__ == "__main__":
    print("Running the server in development mode.")
    cPORT = os.getenv("PORT", 82)
    app.run(port = cPORT, host = "0.0.0.0")