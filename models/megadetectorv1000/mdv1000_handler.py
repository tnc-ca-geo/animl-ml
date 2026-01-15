"""TorchServe handler for MegaDetector v1000 using the megadetector package."""
from ts.torch_handler.base_handler import BaseHandler
from megadetector.detection.run_detector import load_detector
from megadetector.visualization import visualization_utils as vis_utils
import io
import base64
import torch
import torchvision


class ModelHandler(BaseHandler):
    """Handler for MegaDetector v1000."""
    
    def initialize(self, context):
        """Load the MegaDetector model."""
        # Get model directory from context
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model from local file in the .mar archive
        model_path = f"{model_dir}/md_v1000.0.0-redwood.pt"
        self.model = load_detector(model_path)
        self.initialized = True
    
    def preprocess(self, data):
        """Extract and load image from request."""
        row = data[0]
        image = row.get("data") or row.get("body")
        
        if isinstance(image, str):
            image = base64.b64decode(image)
        
        if isinstance(image, (bytearray, bytes)):
            return vis_utils.load_image(io.BytesIO(image))
        
        raise ValueError("Unsupported image format")
    
    def inference(self, image):
        """Run MegaDetector inference."""
        return self.model.generate_detections_one_image(image)
    
    def postprocess(self, result):
        """Convert MegaDetector output to Animl format with NMS."""
        # Apply NMS to remove overlapping detections
        detections = result['detections']
        
        if len(detections) == 0:
            return [[]]

        for det in detections:
            det['bbox'] = self.xywh2xyxy(det['bbox'])
        
        # Convert to tensors for NMS
        boxes = torch.tensor([[d['bbox'][0], d['bbox'][1], 
                               d['bbox'][0] + d['bbox'][2], 
                               d['bbox'][1] + d['bbox'][3]] for d in detections])
        scores = torch.tensor([d['conf'] for d in detections])
        
        # Apply NMS with IoU threshold of 0.45
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
        
        # Filter detections
        filtered_detections = []
        for idx in keep_indices:
            det = detections[idx]
            filtered_detections.append({
                "x1": det['bbox'][0],
                "y1": det['bbox'][1],
                "x2": det['bbox'][2],
                "y2": det['bbox'][3],
                "confidence": det['conf'],
                "class": int(det['category'])  # MD uses 0-2, Animl uses 1-3
            })
        
        return [filtered_detections]



    def xywh2xyxy(self, bbox):
        return [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
        ]


