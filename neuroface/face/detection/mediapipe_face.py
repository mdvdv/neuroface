import cv2
import numpy as np
import mediapipe
from typing import List


class FaceDetection(object):
    """ MediaPipe Face Detection implementation.
    
    MediaPipe Face Detection processes an RGB image and returns a list of the detected faces.
    
    Example:
        >>> import cv2
        >>> from neuroface import FaceDetection
        >>> face_detector = FaceDetection(model_selection=0, min_detection_confidence=0.9)
        >>> image = cv2.imread('test.jpg')
        >>> face_array = face_mesh.extract(image)
    """
    
    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5
    ) -> None:
        """
        Args:
            model_selection (int, optional): 0 or 1. 0 to select a short-range model that works
                best for faces within 2 meters from the camera, and 1 for a full-range
                model best for faces within 5 meters.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful.
        """
        
        super().__init__()
        
        face_detector = mediapipe.solutions.face_detection
        self.model = face_detector.FaceDetection(model_selection=model_selection,
            min_detection_confidence=min_detection_confidence)
    
    def detect(self, image: np.ndarray) -> List:
        """
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
        """
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        face_array = []
        prediction = self.model.process(image)
        
        if prediction.detections:
            for idx in range(len(prediction.detections)):
                bounding_box = prediction.detections[idx].location_data.relative_bounding_box
                face_location = np.array([[int(bounding_box.xmin * image_width), int(bounding_box.ymin * image_height)],
                                          [int(bounding_box.width * image_width), int(bounding_box.height * image_height)]])
                
                face_array.append(face_location)
        
        return np.array(face_array)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """ Cropping area where the face is located.
        
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
        """
        
        face_batch = []
        face_array = self.detect(image=image)
        
        if face_array is not None:
            for face_location in face_array:
                face_image = image[face_location[0][1]:face_location[0][1]+face_location[1][1],
                                   face_location[0][0]:face_location[0][0]+face_location[1][0]]
                
                face_batch.append(face_image)
        
        return face_batch
