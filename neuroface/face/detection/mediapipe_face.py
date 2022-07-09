import cv2
import numpy as np
import mediapipe


class MediapipeFace(object):
    """ MediaPipe model implementation for face detection.
    
    Args:
        image (np.ndarray): BGR image represented as numpy ndarray.
        model (module): MediaPipe solution Python API.
        model_selection (int): 0 or 1. 0 to select a short-range model that works
            best for faces within 2 meters from the camera, and 1 for a full-range
            model best for faces within 5 meters.
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful.
    """
    
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        super().__init__()
        
        face_detector = mediapipe.solutions.face_detection
        self.model = face_detector.FaceDetection(model_selection=model_selection,
            min_detection_confidence=min_detection_confidence)
    
    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        face_location = np.array([])
        prediction = self.model.process(image)
        
        if prediction.detections:
            bounding_box = prediction.detections[0].location_data.relative_bounding_box
            face_location = np.array([[int(bounding_box.xmin * image_width), int(bounding_box.ymin * image_height)],
                                      [int(bounding_box.width * image_width), int(bounding_box.height * image_height)]])
        
        return face_location
    
    def extract(self, image):
        """ Cropping area where the face is located.
        
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
        """
        
        face_image = np.array([])
        face_location = self.detect(image=image)
        
        if face_location is not None:
            face_image = image[face_location[0][1]:face_location[0][1]+face_location[1][1],
                               face_location[0][0]:face_location[0][0]+face_location[1][0]]
        
        return face_image