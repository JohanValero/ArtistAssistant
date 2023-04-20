from flask import Flask, request, jsonify, send_file
import base64
from io import BytesIO
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from controlnet_aux import OpenposeDetector
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

gSTABLE_DIFUSSION_INITIATED = False
gOBJECT_DETECTION_INITIATED = False
gSAM_INITIATED = False
gOPENPOSE_INITIATED = False

#cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_h", "./resources/models/sam_vit_h_4b8939.pth" # 2.5G
#cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_l", "./resources/models/sam_vit_l_0b3195.pth" # 1.2 GB
cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_b", "./resources/models/sam_vit_b_01ec64.pth" # 300 MB

gSAM_predictor = None
gSAM_model = None
cSAM_DEVICE = "cpu"

gObject_detector_processor = None
gObject_detector_model = None
gOBJECT_DETECTOR_DEVICE = "cuda"

gOpen_pose_detector = None
cOPEN_POSE_DEVICE = "cpu"

def array2image(iImage : np.ndarray, iArrayMode) -> Image:
    return Image.fromarray(iImage, iArrayMode)

@app.route("/")
def home():
  vResponse = {"message": "¡Hola, bienvenido a la aplicación Flask!"}
  return jsonify(vResponse)

@app.route("/init/object_detection", methods=["GET"])
def init_object_detection():
  global gOBJECT_DETECTION_INITIATED
  global gObject_detector_processor
  global gObject_detector_model
  global gOBJECT_DETECTOR_DEVICE

  if gOBJECT_DETECTION_INITIATED:
    return jsonify({"status": 1})
  gObject_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
  gObject_detector_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
  gObject_detector_model.to(device = gOBJECT_DETECTOR_DEVICE)

  gOBJECT_DETECTION_INITIATED = True
  return jsonify({"status": 1})

@app.route("/init/sam", methods = ["GET"])
def init_sam():
  global gSAM_INITIATED
  global gSAM_predictor
  global gSAM_model
  global cSAM_DEVICE

  if gSAM_INITIATED:
    return jsonify({"status": 1})

  gSAM_model = sam_model_registry[cSAM_MODEL_TYPE](checkpoint=cSAM_CHECKPOINT)
  gSAM_model.to(device = cSAM_DEVICE)
  gSAM_predictor = SamPredictor(gSAM_model)
  
  gSAM_INITIATED = True
  return jsonify({"status": 1})

@app.route("/init/openpose", methods = ["GET"])
def init_pose():
  global gOpen_pose_detector
  global gOPENPOSE_INITIATED

  if gOPENPOSE_INITIATED:
    return jsonify({"status": 1})

  gOpen_pose_detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
  gOpen_pose_detector.to(cOPEN_POSE_DEVICE)
  gOPENPOSE_INITIATED = True
  return jsonify({"status": 1})

@app.route("/set/sam", methods = ["POST"])
def set_image_same():
  global gSAM_INITIATED
  global gSAM_predictor

  data = request.get_json()
  vImageBase64 = data['image']
  vImageBase64 = base64.b64decode(vImageBase64)
  vImage = Image.open(BytesIO(vImageBase64))
  vImage = np.array(vImage)
  
  gSAM_predictor.set_image(vImage)
  return jsonify({"status": 1})

@app.route("/genera/sam/box", methods = ["POST"])
def segment_image_sam():
  data = request.get_json()
  vBoxes = data['boxes']
  vShape = np.array(data['shape'])
  vTorchBoxes = torch.tensor(vBoxes, device=gSAM_predictor.device)
  vTorchBoxes = gSAM_predictor.transform.apply_boxes_torch(vTorchBoxes, vShape)
  vMasks, _, _ = gSAM_predictor.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = vTorchBoxes,
      multimask_output = False,
    )
  vMasks = [x.cpu().numpy() for x in vMasks]
  vMasks = [np.packbits(x, axis = -1) for x in vMasks]
  vMasks = [base64.b64encode(x).decode('utf-8') for x in vMasks]

  vRespond = {
    "status": 1,
    "masks": vMasks
  }
  return jsonify(vRespond)

@app.route("/genera/object_detection", methods = ["POST"])
def detect_object():
  global gObject_detector_processor
  global gObject_detector_model
  global gOBJECT_DETECTOR_DEVICE

  data = request.get_json()
  vImageBase64 = data['image']
  vImageBase64 = base64.b64decode(vImageBase64)
  vImage = Image.open(BytesIO(vImageBase64))

  vModelInput = gObject_detector_processor(
      images = vImage,
      return_tensors = "pt"
    ).to(gOBJECT_DETECTOR_DEVICE)
  vModelOutput = gObject_detector_model(**vModelInput)
  
  vTargetSizes = torch.tensor([vImage.size[::-1]])
  vResults = gObject_detector_processor.post_process_object_detection(
    vModelOutput,
    target_sizes = vTargetSizes,
    threshold=0.9
  )[0]
  vResultResponse = []
  for tTorchScore, tTorchLabel, tTorchBox in zip(vResults["scores"], vResults["labels"], vResults["boxes"]):
    vResult = {
        "score": tTorchScore.item(),
        "label": tTorchLabel.item(),
        "label_name": gObject_detector_model.config.id2label[tTorchLabel.item()],
        "box": tTorchBox.tolist()
    }
    vResultResponse.append(vResult)
  vJsonResponse = {
    'result': 'success',
    'data': vResultResponse
  }
  return jsonify(vJsonResponse)

@app.route("/genera/pose_estimation", methods = ['POST'])
def genera_pose():
  data = request.get_json()
  vImageBase64 = data['image']
  vImageBase64 = base64.b64decode(vImageBase64)
  vImage = Image.open(BytesIO(vImageBase64))
  
  # HERE IA MAGIC
  vGeneratedImage = gOpen_pose_detector(vImage)

  vBuffer = BytesIO()
  vGeneratedImage.save(vBuffer, format="JPEG")
  vImageBase64 = base64.b64encode(vBuffer.getvalue()).decode('utf-8')
  image_base64 = vImageBase64

  response = {
    'result': 'success',
    'image': image_base64
  }
  return jsonify(response)

@app.route('/genera/control_net', methods=['POST'])
def genera_controlnet():
  data = request.get_json()
  vImageBase64 = data['image']
  vImageBase64 = base64.b64decode(vImageBase64)
  vImage = Image.open(BytesIO(vImageBase64))
  
  # HERE IA MAGIC
  vGeneratedImage = vImage
  """
  vGeneratedImage = vStablePipe(
      "Beautiful cute red female cyborg, cyberpunk, art by Richard Anderson",
      vImage,
      num_inference_steps = 20
    ).images[0]
  #"""

  vBuffer = BytesIO()
  vGeneratedImage.save(vBuffer, format="JPEG")
  vImageBase64 = base64.b64encode(vBuffer.getvalue()).decode('utf-8')
  image_base64 = vImageBase64

  response = {
    'result': 'success',
    'image': image_base64
  }
  return jsonify(response)

if __name__ == "__main__":
    app.run(debug = True)