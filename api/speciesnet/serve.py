import json
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from speciesnet.scripts.run_server import SpeciesNetLitAPI
from speciesnet import DEFAULT_MODEL

# Initialize FastAPI app
app = FastAPI()

# Initialize SpeciesNet API
model_name = DEFAULT_MODEL
try:
    api = SpeciesNetLitAPI(model_name=model_name, geofence=True)
    api.setup(device=None)  # Initialize the model
    logger.info(f"Initialized SpeciesNet API with model: {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    try:
        if api.model:  # Check if model is loaded
            return JSONResponse(content={"status": "healthy"}, status_code=200)
    except:
        pass
    return JSONResponse(content={"status": "unhealthy"}, status_code=500)

@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker invocation endpoint"""
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
