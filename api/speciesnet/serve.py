# FastAPI server for SpeciesNet model inference

import json
import logging
import os
from typing import Optional, Literal

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import uvicorn
from speciesnet import SpeciesNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize SpeciesNet model
model_name = "/opt/ml/model/speciesnet/models/google/speciesnet/pytorch/v4.0.1a/1"
try:
    model = SpeciesNet(
        model_name=model_name,
        components="all",  # Default to all components
        geofence=True  # Default geofencing enabled
    )
    logger.info(f"Initialized SpeciesNet model: {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    try:
        if model:  # Check if model is loaded
            return JSONResponse(content={"status": "healthy"}, status_code=200)
    except:
        pass
    return JSONResponse(content={"status": "unhealthy"}, status_code=500)

@app.post("/invocations")
async def invoke(
    request: Request,
    components: Optional[Literal["all", "classifier", "detector"]] = Query(
        default="all", description="Model components to run"
    ),
    geofence: bool = Query(
        default=True, description="Whether to enable geofencing"
    ),
    batch_size: int = Query(
        default=8, description="Batch size for classifier inference"
    )
):
    """SageMaker invocation endpoint with extended options"""
    try:
        # Get raw request body
        body = await request.body()
        input_data = json.loads(body)

        if 'image_data' not in input_data:
            return JSONResponse(
                status_code=400,
                content={"error": "Input must contain 'image_data' field"}
            )

        # Create temporary file for the image
        import base64
        from PIL import Image
        from io import BytesIO

        # Decode and save image
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

        # Add optional country parameter
        # TODO: validate this is 3 letter ISO
        if 'country' in input_data:
            instances_dict['instances'][0]['country'] = input_data['country']

        print('instances_dict', instances_dict)
        try:
            # Update geofencing setting
            model.geofence = geofence

            # Run prediction based on components
            if components == "classifier":
                # Get bbox from request or use default [x_min, y_min, width, height]
                bbox = input_data.get('bbox', [0, 0, 1, 1])

                # Create detections dict with bbox
                detections_dict = {
                    temp_path: {
                        "detections": [{
                            "bbox": bbox
                        }]
                    }
                }

                predictions_dict = model.classify(
                    instances_dict=instances_dict,
                    detections_dict=detections_dict,
                    batch_size=batch_size
                )
            elif components == "detector":
                predictions_dict = model.detect(
                    instances_dict=instances_dict
                )
            else:  # all components
                predictions_dict = model.predict(
                    instances_dict=instances_dict,
                    batch_size=batch_size
                )

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if predictions_dict is None:
                raise Exception("No predictions returned from model")

            return JSONResponse(content=predictions_dict)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
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
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
