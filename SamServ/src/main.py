import os
import torch
import base64
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from segment_anything import sam_model_registry, SamPredictor

print("Loading model.")
#cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_h", "./models/sam_vit_h_4b8939.pth" # 2.5G
#cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_l", "./models/sam_vit_l_0b3195.pth" # 1.2 GB
cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_b", "./models/sam_vit_b_01ec64.pth" # 300 MB

cSAM_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
cSAM_PREDITOR = None
cSAM_MODEL = None

cSAM_MODEL = sam_model_registry[cSAM_MODEL_TYPE](checkpoint=cSAM_CHECKPOINT)
cSAM_MODEL.to(device = cSAM_DEVICE)
cSAM_PREDITOR = SamPredictor(cSAM_MODEL)

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

@app.route("/genera/segments/box", methods = ["POST"])
def detect_object():
    try:
        # Obtain the parameters.
        vData = request.get_json()
        vBoxes = vData['boxes']
        vImage = vData['image']
        
        # Decoded the Image in a Numpy array.
        vImage = base64.b64decode(vImage)
        vImage = Image.open(BytesIO(vImage))
        vImage = np.array(vImage)
        
        # Convert the boxes in a corect format.
        vBoxes = torch.tensor(vBoxes, device=cSAM_PREDITOR.device)
        vBoxes = cSAM_PREDITOR.transform.apply_boxes_torch(vBoxes, vImage.shape[:2])

        # Generate the masks.
        cSAM_PREDITOR.set_image(vImage)
        vMasks, _, _ = cSAM_PREDITOR.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = vBoxes,
            multimask_output = False,
        )

        vMasks   = [x.cpu().numpy()    for x in vMasks] # Obtain the masks in the GPU to Numpy Arrays.
        vMasks   = [np.squeeze(x)      for x in vMasks] # Modify the shape (1, Width, Height) to (Width, Height).
        vMasks   = [Image.fromarray(x) for x in vMasks] # Convert the np array to Image format.
        vBuffers = [BytesIO()          for _ in vMasks] # Create the buffers.
        _        = [tMask.save(tBuffer, format="jpeg") for tMask, tBuffer in zip(vMasks, vBuffers)] # Save the Image bytes in the buffers.
        vBuffers = [base64.b64encode(tBuffer.getvalue()).decode('utf-8') for tBuffer in vBuffers  ] # Enconde the byte data in utf-8

        vJsonResponse = {"masks": vBuffers}
        return generate_json_response(vJsonResponse)
    except Exception as ex:
        print(ex)
        raise ex

if __name__ == "__main__":
    print("Running the server in development mode.")
    cPORT = os.getenv("PORT", 80)
    app.run(port = cPORT, host = "0.0.0.0", debug = True)