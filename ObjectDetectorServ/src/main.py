import os
import torch
import base64

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from transformers import DetrImageProcessor, DetrForObjectDetection

print("Loading model.")
cOBJECT_DETECTOR_DEVICE = os.getenv("MODEL_DEVICE", "cuda:0")
cOBJECT_DETECTOR_PROCESSOR = DetrImageProcessor.from_pretrained("./models/object_detector")
cOBJECT_DETECTOR_MODEL = DetrForObjectDetection.from_pretrained("./models/object_detector")
cOBJECT_DETECTOR_MODEL = cOBJECT_DETECTOR_MODEL.to(cOBJECT_DETECTOR_DEVICE)
print(f"Model loaded in device {cOBJECT_DETECTOR_DEVICE}.")

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

@app.route("/genera/object_detection", methods = ["POST"])
def detect_object():
    try:
        vRequestJson = request.get_json()
        vImageBase64 = vRequestJson['image']
        vImageBase64 = base64.b64decode(vImageBase64)
        vImage = Image.open(BytesIO(vImageBase64))

        vModelInput = cOBJECT_DETECTOR_PROCESSOR(
                images = vImage,
                return_tensors = "pt"
            ).to(cOBJECT_DETECTOR_DEVICE)
        vModelOutput = cOBJECT_DETECTOR_MODEL(**vModelInput)
    
        vTargetSizes = torch.tensor([vImage.size[::-1]])
        vResults = cOBJECT_DETECTOR_PROCESSOR.post_process_object_detection(
                vModelOutput,
                target_sizes = vTargetSizes,
                threshold = 0.9
            )[0]
        vResultResponse = []
        for tTorchScore, tTorchLabel, tTorchBox in zip(vResults["scores"], vResults["labels"], vResults["boxes"]):
            vResult = {
                "score": tTorchScore.item(),
                "label": tTorchLabel.item(),
                "label_name": cOBJECT_DETECTOR_MODEL.config.id2label[tTorchLabel.item()],
                "box": tTorchBox.tolist()
            }
            vResultResponse.append(vResult)
        vJsonResponse = {
            'result': 'success',
            'data': vResultResponse
        }
        return generate_json_response(vJsonResponse)
    except Exception as ex:
        print(ex)
        raise ex

if __name__ == "__main__":
    print("Running the server in development mode.")
    cPORT = os.getenv("PORT", 88)
    app.run(port = cPORT, host = "0.0.0.0")