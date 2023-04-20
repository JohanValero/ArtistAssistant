import os
import torch
import base64
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from controlnet_aux import OpenposeDetector

print("Loading model.")
cOPEN_POSE_DEVICE = "cuda"
gOPEN_POSE_DETECTOR = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
#gOPEN_POSE_DETECTOR.to(cOPEN_POSE_DEVICE)

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

@app.route("/genera/open_pose", methods = ["POST"])
def detect_object():
    try:
        # Obtain the parameters.
        vData = request.get_json()
        vImage = vData['image']
        
        # Decoded the Image in a Numpy array.
        vImage = base64.b64decode(vImage)
        vImage = Image.open(BytesIO(vImage))
        #vImage = np.array(vImage)

        vGeneratedImage = gOPEN_POSE_DETECTOR(vImage)

        vBuffer = BytesIO()
        vGeneratedImage.save(vBuffer, format="JPEG")
        vImageBase64 = base64.b64encode(vBuffer.getvalue()).decode('utf-8')
        image_base64 = vImageBase64

        vResponse = {
            'result': 'success',
            'image': image_base64
        }
        return generate_json_response(vResponse)
    except Exception as ex:
        print(ex)
        raise ex

if __name__ == "__main__":
    print("Running the server in development mode.")
    cPORT = os.getenv("PORT", 81)
    app.run(port = cPORT, host = "0.0.0.0", debug = True)