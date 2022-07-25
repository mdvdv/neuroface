import cv2
import numpy as np
import mediapipe
from typing import Tuple


class Hands(object):
    """ MediaPipe Hands implementation.
    
    MediaPipe Hands processes an RGB image and returns 21 hand landmarks and handedness of each detected hand.
    
    Example:
        >>> import cv2
        >>> from neuroface import Hands
        >>> model = Hands(max_num_hands=3, min_detection_confidence=0.75)
        >>> image = cv2.imread('test.jpg')
        >>> hands_array, hands_pred = model.detect(image, classify=True)
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """
        Args:
            static_image_mode (int, optional): Whether to treat the input images as a batch
                of static and possibly unrelated images, or a video stream.
            max_num_hands (int, optional): Maximum number of hands to detect.
            model_complexity (int, optional): Complexity of the hand landmark model: 0 or 1.
                Landmark accuracy as well as inference latency generally go up with the model complexity.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for hand
                detection to be considered successful.
            min_tracking_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for the
                hand landmarks to be considered tracked successfully.
        """
        
        super().__init__()
        
        hands_detector = mediapipe.solutions.hands
        self.model = hands_detector.Hands(static_image_mode=static_image_mode,
            max_num_hands=max_num_hands, model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    
    def detect(self, image: np.ndarray, normalize: bool = False, classify: bool = False) -> "np.ndarray | Tuple[np.ndarray, np.ndarray]":
        """
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
            normalize (bool, optional): Apply minimax normalization to hand landmarks.
            classify (bool, optional): Whether or not to return the handedness prediction.
                Returns 0 or 1, where 0 is left hand, 1 is right hand.
        """
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        hands_array = []
        hands_pred = []
        prediction = self.model.process(image)
        
        if prediction.multi_hand_landmarks:
            for idx in range(len(prediction.multi_hand_landmarks)):
                landmarks = prediction.multi_hand_landmarks[idx].landmark
                hands_vector = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                
                if normalize:
                    # Normalization and centering of the hands from -0.5 to 0.5 in width, height and depth of the hands.
                    min_value = hands_vector.min(axis=0)
                    max_value = hands_vector.max(axis=0)
                    
                    hands_vector = np.absolute(hands_vector - min_value) / np.absolute(max_value - min_value) - 0.5
                
                if classify:
                    hand_class = prediction.multi_handedness[idx].classification[0].index
                    hands_pred.append(hand_class)
                
                hands_array.append(hands_vector)
        
        if classify:
            return np.array(hands_array), np.array(hands_pred)
        
        return np.array(hands_array)