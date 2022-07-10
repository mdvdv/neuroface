import cv2
import numpy as np
import mediapipe


class MediapipeMesh(object):
    """ MediaPipe Face Mesh implementation for face mesh detection.
    
    Args:
        image (np.ndarray): BGR image represented as numpy ndarray.
        model (module): MediaPipe solution Python API.
        static_image_mode (int): Whether to treat the input images as a batch
            of static and possibly unrelated images, or a video stream.
        max_num_faces (int): Maximum number of faces to detect.
            refine_landmarks (bool): Whether to further refine the landmark coordinates
            around the eyes and lips, and output additional landmarks around the irises.
        min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful.
        min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the
            face landmarks to be considered tracked successfully.
        normalize (bool): Apply minimax normalization to facial landmark.
    """
    
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        super().__init__()
        
        mesh_detector = mediapipe.solutions.face_mesh
        self.model = mesh_detector.FaceMesh(static_image_mode=static_image_mode,
            max_num_faces=max_num_faces, refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    
    def detect(self, image, normalize=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_vector = np.array([])
        prediction = self.model.process(image)
        
        if prediction.multi_face_landmarks:
            landmarks = prediction.multi_face_landmarks[0].landmark
            face_vector = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
            
            if normalize:
                # Normalization and centering of the face from -0.5 to 0.5 in width, height and depth of the face.
                min_value = face_vector.min(axis=0)
                max_value = face_vector.max(axis=0)
                
                face_vector = np.absolute(face_vector - min_value) / np.absolute(max_value - min_value) - 0.5
        
        return face_vector