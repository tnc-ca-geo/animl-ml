import base64
import requests
import json
API_URL = 'http://localhost:8501/v1/models/mira-large:classify'
IMG_PATH = '/home/rave/animl/animl-ml/input/sample-img-fox.jpg'
# Open and read image as bitstring
input_image = open(IMG_PATH, "rb").read()
print("Raw bitstring: " + str(input_image[:10]) + " ... " + str(input_image[-10:]))

# Encode image in b64
encoded_input_string = base64.b64encode(input_image)
input_string = encoded_input_string.decode("utf-8")
print("Base64 encoded string: " + input_string[:10] + " ... " + input_string[-10:])
predict_request = '{"instances" : [{"b64": "%s"}]}' % input_string
response = requests.post(API_URL, data=predict_request)
print(response.json())
