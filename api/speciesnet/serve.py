# FastAPI server that wraps around the SpeciesNet Litserve Class

import json
import logging
import os
import threading
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from speciesnet.scripts.run_server import SpeciesNetLitAPI
from speciesnet import DEFAULT_MODEL

class ModelLoader:
    def __init__(self):
        self.model = None
        self.error = None
        self._loading = False
        self._lock = threading.Lock()

    def start_loading(self):
        if not self._loading:
            self._loading = True
            thread = threading.Thread(target=self._load_model)
            thread.daemon = True
            thread.start()

    def _load_model(self):
        try:
            api = SpeciesNetLitAPI(model_name=DEFAULT_MODEL, geofence=True)
            api.setup(device=None)
            with self._lock:
                self.model = api
            logger.info(f"Initialized SpeciesNet API with model: {DEFAULT_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.error = str(e)
        finally:
            self._loading = False

    def is_ready(self) -> bool:
        return self.model is not None

    def get_model(self) -> Optional[SpeciesNetLitAPI]:
        return self.model

    def get_error(self) -> Optional[str]:
        return self.error

# Initialize FastAPI app
app = FastAPI()

# Initialize model loader
model_loader = ModelLoader()
model_loader.start_loading()

@app.get("/ping")
async def ping():
    """Health check endpoint that returns healthy even while model is loading"""
    if model_loader.error:
        return JSONResponse(content={"status": "unhealthy", "error": model_loader.get_error()}, status_code=500)
    return JSONResponse(content={"status": "healthy"}, status_code=200)

@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker invocation endpoint"""
    # Check if model is ready
    if not model_loader.is_ready():
        if model_loader.error:
            return JSONResponse(
                status_code=500,
                content={"error": f"Model failed to load: {model_loader.get_error()}"}
            )
        return JSONResponse(
            status_code=503,
            content={"error": "Model is still loading"}
        )

    try:
        # Get raw request body
        body = await request.body()
        input_data = json.loads(body)
        
        # Convert SageMaker format to SpeciesNet format
        if 'image_data' in input_data:
            # Create a temporary file for the image
            import base64
            from PIL import Image
            from io import BytesIO
            
            image_bytes = base64.b64decode(input_data['image_data'])
            image = Image.open(BytesIO(image_bytes))
            
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
            # Create instances dict
            instances_dict = {
                "instances": [{
                    "filepath": temp_path
                }]
            }
            
            # Add optional parameters
            if 'country' in input_data:
                # TODO: validate that this is 3 letter ISO code
                instances_dict['instances'][0]['country'] = input_data['country']
                
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Input must contain 'image_data' field"}
            )
            
        # Use the SpeciesNet API directly
        try:
            api = model_loader.get_model()
            # Decode request
            decoded_request = api.decode_request(instances_dict, context=None)
            
            # Run prediction
            result = api.predict(decoded_request, context=None)
            
            # Encode response
            response = api.encode_response(result, context=None)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return response
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Model error: {str(e)}"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def main():
    """Run the server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
