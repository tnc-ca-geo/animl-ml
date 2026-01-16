"""TorchServe handler for MegaDetector v1000 using the megadetector package."""

from ts.torch_handler.base_handler import BaseHandler
from megadetector.detection.run_detector import load_detector
from megadetector.visualization import visualization_utils as vis_utils
import io
import base64

NMS_THRESHOLD = 0.0005
DETECTION_THRESHOLD = 0.4


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
        """Run MegaDetector inference with low threshold for NMS."""
        return self.model.generate_detections_one_image(
            image, detection_threshold=NMS_THRESHOLD
        )

    def postprocess(self, result):
        """Convert MegaDetector output to Animl format with two-stage filtering."""
        detections = result["detections"]

        if len(detections) == 0:
            return [[]]

        # Filter by high confidence threshold for final output
        filtered_detections = []
        for det in detections:
            if det["conf"] > DETECTION_THRESHOLD:
                # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]
                bbox = det["bbox"]
                filtered_detections.append(
                    {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[0] + bbox[2],
                        "y2": bbox[1] + bbox[3],
                        "confidence": det["conf"],
                        "class": int(det["category"]),
                    }
                )

        return [filtered_detections]
