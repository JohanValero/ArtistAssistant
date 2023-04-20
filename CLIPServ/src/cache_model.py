from transformers import DetrImageProcessor, DetrForObjectDetection

gObject_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
gObject_detector_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

gObject_detector_processor.save_pretrained("./models/object_detector")
gObject_detector_model.save_pretrained("./models/object_detector")