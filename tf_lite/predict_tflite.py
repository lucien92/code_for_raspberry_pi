#import json
import cv2
import numpy as np
from utils_lite import draw_boxes, decode_netout
import tflite_runtime.interpreter as tflite


config_path = '/home/lucien/project_ornithoScope_lucien/src/config/benchmark_config/model_classic.json'
weights_path = '/home/lucien/project_ornithoScope_lucien/src/data/saved_weights/benchmark_weights/model_classic_bestLoss.h5'
lite_path = '/home/lucien/project_ornithoScope_lucien/src/tf_lite/classic_tf_lite.tflite'
anchors_list = [5.49950,8.57597, 9.26930,17.66783, 10.56113,10.43321, 15.44298,23.17856, 34.00303,34.41259]
labels = ["MESCHA", "SITTOR", "MESBLE", "MESNON", "PINARB", "ACCMOU", "ROUGOR", "VEREUR", "TOUTUR", "ECUROU", "PIEBAV", "MULGRI", "MESNOI", "MESHUP"]



def resize(image, input_size = (224,224), gray_mode = False):
    if len(image.shape) == 3 and gray_mode:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
    elif len(image.shape) == 2 and not gray_mode:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]

    image = cv2.resize(image, (input_size[1], input_size[0]))
    image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

    '''devrait-on garder la fonction ci-dessous?'''
    #image = feature_extractor.normalize(image) #utilise le modèle tesnorflow pour normaliser l'image
    image = (image/255.0) - 1.0
    if len(image.shape) == 3:
        input_image = image[np.newaxis, :]
    else:
        input_image = image[np.newaxis, ..., np.newaxis]
    
    return input_image

def predict(interpreter, image, labels, iou_threshold=0.5, score_threshold=0.3):

    anchors_list = [5.49950,8.57597, 9.26930,17.66783, 10.56113,10.43321, 15.44298,23.17856, 34.00303,34.41259]
    
    input_image = resize(image)

    # Extract details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_type = input_details[0]['dtype']

    # Convert frame to input type
    input_image = input_image.astype(input_type)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    netout = interpreter.get_tensor(output_details[0]['index'])[0]
    anchors = anchors_list
    nb_class = len(labels)
    boxes = decode_netout(netout, anchors, nb_class, score_threshold, iou_threshold) 

    return boxes


# One image
image_path = '/home/lucien/project_ornithoScope_lucien/src/tf_lite/test2.jpg'

# Open image
frame = cv2.imread(image_path)

# Predict
#boxes = yolo.predict(frame) #utilise seulement le modèle tflite

interpreter = tflite.Interpreter(model_path=lite_path)
interpreter.allocate_tensors()

boxes = predict(interpreter, frame, labels) #utilise le modèle tflite et le modèle tensorflow

# Draw boxes
frame = draw_boxes(frame, boxes, labels) #draw_boxes n'utilise pas tf
# Write image output
cv2.imwrite(image_path[:-4] + '_lite_detected.jpg', frame)


