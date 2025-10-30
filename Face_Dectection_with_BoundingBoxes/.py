import cv2
import numpy as np
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceData:
    """Data class for face detection results."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    face_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, window_size=30):
        self.fps_history = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self, detection_time):
        """Update performance metrics."""
        self.frame_count += 1
        self.detection_times.append(detection_time)
        
        if len(self.detection_times) > 1:
            fps = 1.0 / np.mean(self.detection_times)
            self.fps_history.append(fps)
    
    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        return {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'frames_processed': self.frame_count,
            'total_time': time.time() - self.start_time
        }


class FacialLandmarkDetector:
    """Detect facial landmarks (68 points)."""
    
    def __init__(self):
        self.detector = None
        self.predictor = None
        self._load_models()
    
    def _load_models(self):
        """Load dlib models for landmark detection."""
        try:
            import dlib
            
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            
            if os.path.exists(predictor_path):
                self.predictor = dlib.shape_predictor(predictor_path)
                logger.info("✓ Loaded facial landmark detector")
            else:
                logger.warning("⚠ Facial landmark model not found. Download from:")
                logger.warning("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        except ImportError:
            logger.warning("⚠ dlib not installed. Landmark detection disabled.")
    
    def detect_landmarks(self, image, bbox):
        """Detect 68 facial landmarks."""
        if not self.predictor:
            return None
        
        import dlib
        
        x, y, w, h = bbox
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        shape = self.predictor(image, rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        return landmarks


class AgeGenderDetector:
    """Detect age and gender using deep learning."""
    
    def __init__(self):
        self.age_net = None
        self.gender_net = None
        self._load_models()
    
    def _load_models(self):
        """Load age and gender detection models."""
        age_proto = "age_deploy.prototxt"
        age_model = "age_net.caffemodel"
        gender_proto = "gender_deploy.prototxt"
        gender_model = "gender_net.caffemodel"
        
        if all(os.path.exists(f) for f in [age_proto, age_model, gender_proto, gender_model]):
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
            logger.info("✓ Loaded age/gender detection models")
        else:
            logger.warning("⚠ Age/gender models not found")
    
    def predict(self, face_img):
        """Predict age and gender."""
        if not self.age_net or not self.gender_net:
            return None, None
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        
        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
        
        # Age prediction
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                   '(38-43)', '(48-53)', '(60-100)']
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = age_list[age_preds[0].argmax()]
        
        return age, gender


class EmotionDetector:
    """Detect facial emotions using deep learning."""
    
    def __init__(self):
        self.model = None
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self._load_model()
    
    def _load_model(self):
        """Load emotion detection model."""
        model_path = "emotion_model.h5"
        
        if os.path.exists(model_path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(model_path)
                logger.info("✓ Loaded emotion detection model")
            except ImportError:
                logger.warning("⚠ TensorFlow not installed. Emotion detection disabled.")
        else:
            logger.warning("⚠ Emotion model not found")
    
    def predict(self, face_img):
        """Predict emotion from face."""
        if not self.model:
            return None
        
        # Preprocess
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)
        
        # Predict
        prediction = self.model.predict(reshaped, verbose=0)
        emotion_idx = np.argmax(prediction)
        
        return self.emotions[emotion_idx]


class FaceRecognizer:
    """Face recognition using embeddings."""
    
    def __init__(self):
        self.model = None
        self.known_faces = {}
        self.face_counter = 0
        self._load_model()
    
    def _load_model(self):
        """Load face recognition model."""
        try:
            import face_recognition
            self.model = face_recognition
            logger.info("✓ Loaded face recognition model")
        except ImportError:
            logger.warning("⚠ face_recognition not installed. Recognition disabled.")
    
    def get_embedding(self, image, bbox):
        """Get face embedding vector."""
        if not self.model:
            return None
        
        x, y, w, h = bbox
        face_img = image[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Get encoding
        encodings = self.model.face_encodings(rgb)
        
        return encodings[0] if encodings else None
    
    def recognize_face(self, embedding, threshold=0.6):
        """Recognize face from embedding."""
        if embedding is None or not self.known_faces:
            return None
        
        min_distance = float('inf')
        best_match = None
        
        for face_id, known_embedding in self.known_faces.items():
            distance = np.linalg.norm(embedding - known_embedding)
            
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match = face_id
        
        return best_match
    
    def add_face(self, embedding):
        """Add new face to database."""
        if embedding is None:
            return None
        
        self.face_counter += 1
        face_id = self.face_counter
        self.known_faces[face_id] = embedding
        
        return face_id


class AdvancedFaceDetector:
    """Advanced face detector with ensemble of models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detectors = {}
        self.landmark_detector = None
        self.age_gender_detector = None
        self.emotion_detector = None
        self.face_recognizer = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize all detection models."""
        # Primary detectors
        if self.config.get('use_haar', False):
            self._load_haar()
        
        if self.config.get('use_dnn', True):
            self._load_dnn()
        
        if self.config.get('use_mtcnn', False):
            self._load_mtcnn()
        
        # Advanced features
        if self.config.get('detect_landmarks', False):
            self.landmark_detector = FacialLandmarkDetector()
        
        if self.config.get('detect_age_gender', False):
            self.age_gender_detector = AgeGenderDetector()
        
        if self.config.get('detect_emotion', False):
            self.emotion_detector = EmotionDetector()
        
        if self.config.get('face_recognition', False):
            self.face_recognizer = FaceRecognizer()
    
    def _load_haar(self):
        """Load Haar Cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade_path)
        
        if not detector.empty():
            self.detectors['haar'] = detector
            logger.info("✓ Loaded Haar Cascade detector")
    
    def _load_dnn(self):
        """Load DNN face detector."""
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"
        
        if os.path.exists(model_file) and os.path.exists(config_file):
            self.detectors['dnn'] = cv2.dnn.readNetFromCaffe(config_file, model_file)
            logger.info("✓ Loaded DNN detector")
        else:
            logger.warning("⚠ DNN model files not found")
    
    def _load_mtcnn(self):
        """Load MTCNN detector."""
        try:
            from mtcnn import MTCNN
            self.detectors['mtcnn'] = MTCNN()
            logger.info("✓ Loaded MTCNN detector")
        except ImportError:
            logger.warning("⚠ MTCNN not installed")
    
    def _detect_dnn(self, image, confidence_threshold):
        """Detect faces using DNN."""
        if 'dnn' not in self.detectors:
            return []
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.detectors['dnn'].setInput(blob)
        detections = self.detectors['dnn'].forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                x = max(0, x1)
                y = max(0, y1)
                w_box = min(w - x, x2 - x1)
                h_box = min(h - y, y2 - y1)
                
                faces.append(((x, y, w_box, h_box), float(confidence)))
        
        return faces
    
    def _detect_haar(self, image):
        """Detect faces using Haar Cascade."""
        if 'haar' not in self.detectors:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detectors['haar'].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return [((x, y, w, h), 1.0) for (x, y, w, h) in detections]
    
    def _detect_mtcnn(self, image):
        """Detect faces using MTCNN."""
        if 'mtcnn' not in self.detectors:
            return []
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detectors['mtcnn'].detect_faces(rgb)
        
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            faces.append(((x, y, w, h), confidence))
        
        return faces
    
    def _ensemble_detection(self, image, confidence_threshold):
        """Combine results from multiple detectors."""
        all_faces = []
        
        # Run all available detectors
        if 'dnn' in self.detectors:
            all_faces.extend(self._detect_dnn(image, confidence_threshold))
        
        if 'haar' in self.detectors:
            all_faces.extend(self._detect_haar(image))
        
        if 'mtcnn' in self.detectors:
            all_faces.extend(self._detect_mtcnn(image))
        
        # Apply Non-Maximum Suppression
        if all_faces:
            boxes = np.array([face[0] for face in all_faces])
            confidences = np.array([face[1] for face in all_faces])
            
            # Convert to (x1, y1, x2, y2) format
            boxes_nms = np.column_stack([
                boxes[:, 0], boxes[:, 1],
                boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
            ])
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(), confidences.tolist(),
                confidence_threshold, 0.4
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [(boxes[i], confidences[i]) for i in indices]
        
        return []
    
    def detect_faces(self, image, confidence_threshold=0.5) -> List[FaceData]:
        """
        Detect faces and extract all features.
        
        Returns:
            List of FaceData objects with all detected information
        """
        # Get face detections
        detections = self._ensemble_detection(image, confidence_threshold)
        
        face_data_list = []
        
        for bbox, conf in detections:
            face_data = FaceData(bbox=bbox, confidence=conf)
            
            x, y, w, h = bbox
            face_img = image[y:y+h, x:x+w]
            
            # Detect landmarks
            if self.landmark_detector and self.landmark_detector.predictor:
                face_data.landmarks = self.landmark_detector.detect_landmarks(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), bbox
                )
            
            # Detect age/gender
            if self.age_gender_detector and self.age_gender_detector.age_net:
                age, gender = self.age_gender_detector.predict(face_img)
                face_data.age = age
                face_data.gender = gender
            
            # Detect emotion
            if self.emotion_detector and self.emotion_detector.model:
                face_data.emotion = self.emotion_detector.predict(face_img)
            
            # Face recognition
            if self.face_recognizer and self.face_recognizer.model:
                embedding = self.face_recognizer.get_embedding(image, bbox)
                face_data.embedding = embedding
                
                # Try to recognize
                face_id = self.face_recognizer.recognize_face(embedding)
                
                if face_id is None and self.config.get('auto_register', False):
                    face_id = self.face_recognizer.add_face(embedding)
                
                face_data.face_id = face_id
            
            face_data_list.append(face_data)
        
        return face_data_list


class VisualizationEngine:
    """Advanced visualization for face detection results."""
    
    @staticmethod
    def draw_face_data(image, face_data_list: List[FaceData], 
                      show_landmarks=True, show_info=True):
        """Draw comprehensive face visualization."""
        output = image.copy()
        
        for i, face_data in enumerate(face_data_list):
            x, y, w, h = face_data.bbox
            
            # Draw bounding box with gradient effect
            color = (0, 255, 0)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw corners for modern look
            corner_length = 15
            thickness = 3
            
            # Top-left
            cv2.line(output, (x, y), (x + corner_length, y), color, thickness)
            cv2.line(output, (x, y), (x, y + corner_length), color, thickness)
            
            # Top-right
            cv2.line(output, (x + w, y), (x + w - corner_length, y), color, thickness)
            cv2.line(output, (x + w, y), (x + w, y + corner_length), color, thickness)
            
            # Bottom-left
            cv2.line(output, (x, y + h), (x + corner_length, y + h), color, thickness)
            cv2.line(output, (x, y + h), (x, y + h - corner_length), color, thickness)
            
            # Bottom-right
            cv2.line(output, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
            cv2.line(output, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
            
            # Draw landmarks
            if show_landmarks and face_data.landmarks:
                for (lx, ly) in face_data.landmarks:
                    cv2.circle(output, (lx, ly), 1, (0, 255, 255), -1)
            
            # Draw info panel
            if show_info:
                info_lines = []
                
                if face_data.face_id:
                    info_lines.append(f"ID: {face_data.face_id}")
                
                info_lines.append(f"Conf: {face_data.confidence:.2%}")
                
                if face_data.gender:
                    info_lines.append(f"{face_data.gender}, {face_data.age}")
                
                if face_data.emotion:
                    info_lines.append(f"Emotion: {face_data.emotion}")
                
                # Draw semi-transparent panel
                panel_height = len(info_lines) * 25 + 10
                panel_y = max(10, y - panel_height - 10)
                
                overlay = output.copy()
                cv2.rectangle(overlay, (x, panel_y), 
                            (x + max(200, w), panel_y + panel_height),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
                
                # Draw text
                for idx, line in enumerate(info_lines):
                    text_y = panel_y + 20 + (idx * 25)
                    cv2.putText(output, line, (x + 5, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    @staticmethod
    def draw_performance(image, perf_monitor: PerformanceMonitor):
        """Draw performance statistics."""
        stats = perf_monitor.get_stats()
        
        # Draw performance panel
        panel_height = 100
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw stats
        cv2.putText(image, f"FPS: {stats['fps']:.1f}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Detection: {stats['avg_detection_time']*1000:.1f}ms", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Frames: {stats['frames_processed']}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image


class VideoProcessor:
    """Process video with advanced face detection."""
    
    def __init__(self, detector: AdvancedFaceDetector, config: Dict):
        self.detector = detector
        self.config = config
        self.perf_monitor = PerformanceMonitor()
        self.viz_engine = VisualizationEngine()
    
    def process_video(self, source, output_path=None):
        """Process video stream."""
        cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
        
        if not cap.isOpened():
            logger.error("Failed to open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps")
        
        # Setup writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info("Processing video... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Detect faces
                faces = self.detector.detect_faces(
                    frame, 
                    self.config.get('confidence_threshold', 0.5)
                )
                
                detection_time = time.time() - start_time
                self.perf_monitor.update(detection_time)
                
                # Visualize
                output = self.viz_engine.draw_face_data(
                    frame, faces,
                    show_landmarks=self.config.get('show_landmarks', True),
                    show_info=True
                )
                
                output = self.viz_engine.draw_performance(output, self.perf_monitor)
                
                # Write/display
                if writer:
                    writer.write(output)
                
                cv2.imshow("Advanced Face Detection", output)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            stats = self.perf_monitor.get_stats()
            logger.info(f"\nFinal Statistics:")
            logger.info(f"  Average FPS: {stats['fps']:.1f}")
            logger.info(f"  Frames processed: {stats['frames_processed']}")
            logger.info(f"  Total time: {stats['total_time']:.1f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Face Detection and Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-i", "--image", help="Input image path")
    parser.add_argument("-v", "--video", help="Video path or camera index")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--haar", action="store_true", help="Use Haar Cascade")
    parser.add_argument("--dnn", action="store_true", help="Use DNN detector")
    parser.add_argument("--mtcnn", action="store_true", help="Use MTCNN")
    parser.add_argument("--landmarks", action="store_true", help="Detect landmarks")
    parser.add_argument("--age-gender", action="store_true", help="Detect age/gender")
    parser.add_argument("--emotion", action="store_true", help="Detect emotion")
    parser.add_argument("--recognition", action="store_true", help="Enable face recognition")
    parser.add_argument("--auto-register", action="store_true", 
                       help="Auto-register unknown faces")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'use_haar': args.haar,
        'use_dnn': args.dnn or not (args.haar or args.mtcnn),
        'use_mtcnn': args.mtcnn,
        'detect_landmarks': args.landmarks,
        'detect_age_gender': args.age_gender,
        'detect_emotion': args.emotion,
        'face_recognition': args.recognition,
        'auto_register': args.auto_register,
        'confidence_threshold': args.confidence,
        'show_landmarks': args.landmarks
    }
    
    print("\n" + "="*70)
    print("Advanced Face Detection and Analysis System")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = AdvancedFaceDetector(config)
    
    if args.video:
        processor = VideoProcessor(detector, config)
        processor.process_video(args.video, args.output)
    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Failed to load image: {args.image}")
            return
        
        faces = detector.detect_faces(image, config['confidence_threshold'])
        
        viz = VisualizationEngine()
        output = viz.draw_face_data(image, faces, 
                                    show_landmarks=config['show_landmarks'])
        
        if args.output:
            cv2.imwrite(args.output, output)
            logger.info(f"Saved to: {args.output}")
        
        cv2.imshow("Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
