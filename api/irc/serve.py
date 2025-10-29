# FastAPI server for SpeciesNet model inference

import json
import logging
import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

TF_SERVING_URL = "http://localhost:8501/v1/models/irc:predict"

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    logger.info("ping received")
    return JSONResponse(content={"status": "healthy"}, status_code=200)

@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker invocation endpoint with extended options"""
    logger.info("invoke received")
    try:
        # Get raw request body
        body = await request.body()
        input_data = json.loads(body)

        # Validate required fields and set defaults
        if 'image_data' not in input_data:
            return JSONResponse(
                status_code=400,
                content={"error": "Input must contain 'image_data' field"}
            )
        
        # print key input parameters
        logger.info('input_data keys: %s', input_data.keys())

        # # Get  parameters with defaults
        # components = input_data.get('components', 'all')
        # if components not in ['all', 'classifier', 'detector']:
        #     return JSONResponse(
        #         status_code=400,
        #         content={"error": "components must be one of: all, classifier, detector"}
        #     )

        # geofence = input_data.get('geofence', True)
        # batch_size = input_data.get('batch_size', 8)

        # # Validate batch_size is integer
        # if not isinstance(batch_size, int):
        #     return JSONResponse(
        #         status_code=400,
        #         content={"error": "batch_size must be an integer"}
        #     )

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

        # # Add  country parameter
        # # TODO: validate this is 3 letter ISO
        # if 'country' in input_data:
        #     instances_dict['instances'][0]['country'] = input_data['country']

        # # Add admin1_region parameter
        # if 'admin1_region' in input_data:
        #     instances_dict['instances'][0]['admin1_region'] = input_data['admin1_region']

        print('instances_dict', instances_dict)
        try:
            # # Update geofencing setting
            # model.geofence = geofence

            # # Run prediction based on components
            # if components == "classifier":
            #     # Get bbox from request or use default [x_min, y_min, width, height]
            #     bbox = input_data.get('bbox', [0, 0, 1, 1])

            #     # Create detections dict with bbox
            #     detections_dict = {
            #         temp_path: {
            #             "detections": [{
            #                 "bbox": bbox
            #             }]
            #         }
            #     }

            #     predictions_dict = model.classify(
            #         instances_dict=instances_dict,
            #         detections_dict=detections_dict,
            #         batch_size=batch_size
            #     )
            # elif components == "detector":
            #     predictions_dict = model.detect(
            #         instances_dict=instances_dict
            #     )
            # else:  # all components
            #     predictions_dict = model.predict(
            #         instances_dict=instances_dict,
            #         batch_size=batch_size
            #     )


            # Forward to TensorFlow Serving
            tf_response = requests.post(TF_SERVING_URL, json=instances_dict)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if tf_response is None:
                raise Exception("No predictions returned from model")

            return JSONResponse(content=tf_response.json(), status_code=tf_response.status_code)
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
