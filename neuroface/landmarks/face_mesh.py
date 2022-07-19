import cv2
import numpy as np
import mediapipe


class FaceMesh(object):
    """ MediaPipe Face Mesh implementation.
    
    MediaPipe Face Mesh processes an RGB image and returns 478 face landmarks on each detected face.
    
    Example:
        >>> import cv2
        >>> from neuroface import FaceMesh
        >>> face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1)
        >>> image = cv2.imread('test.jpg')
        >>> face_array = face_mesh.detect(image)
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        """
        Args:
            static_image_mode (int, optional): Whether to treat the input images as a batch
                of static and possibly unrelated images, or a video stream.
            max_num_faces (int, optional): Maximum number of faces to detect.
            refine_landmarks (bool, optional): Whether to further refine the landmark coordinates
                around the eyes and lips, and output additional landmarks around the irises.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful.
            min_tracking_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for the
                face landmarks to be considered tracked successfully.
        """
        
        super().__init__()
        
        mesh_detector = mediapipe.solutions.face_mesh
        self.model = mesh_detector.FaceMesh(static_image_mode=static_image_mode,
            max_num_faces=max_num_faces, refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    
    def detect(self, image: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
            normalize (bool, optional): Apply minimax normalization to facial landmarks.
        """
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_array = []
        prediction = self.model.process(image)
        
        if prediction.multi_face_landmarks:
            for idx in range(len(prediction.multi_face_landmarks)):
                landmarks = prediction.multi_face_landmarks[idx].landmark
                face_vector = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                
                if normalize:
                    # Normalization and centering of the face from -0.5 to 0.5 in width, height and depth of the face.
                    min_value = face_vector.min(axis=0)
                    max_value = face_vector.max(axis=0)
                    
                    face_vector = np.absolute(face_vector - min_value) / np.absolute(max_value - min_value) - 0.5
                
                face_array.append(face_vector)
        
        return np.array(face_array)
