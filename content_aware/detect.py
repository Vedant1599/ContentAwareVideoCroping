# from model import get_output_layers, draw_prediction
import cv2
import numpy as np

CONFIG_PATH = 'yolov3.cfg'
WEIGHTS_PATH = 'yolov3.weights'
CLASSES_PATH = 'yolov3.txt'

class detect():
    def __init__(self):
        # Load YOLO model and classes
        self.net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

        # Load class names
        with open(CLASSES_PATH, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers
    
    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # label = str(classes[class_id])
        # color = (255, 0, 0)  # You can customize the color here if needed
        # cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pass
    
    def detect_objects(self, image):
        # Get frame dimensions
        Width = image.shape[1]
        Height = image.shape[0]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Set blob as input to the network
        self.net.setInput(blob)

        # Forward pass to get output layers
        outs = self.net.forward(self.get_output_layers())

        # Lists to store detected objects
        class_ids = []
        confidences = []
        boxes = []

        # Confidence and NMS thresholds
        conf_threshold = 0.995
        nms_threshold = 1

        # Post-processing
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Draw bounding boxes and labels
        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        return boxes
    