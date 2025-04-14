# SpeciesNet

## Running the model locally

1. Build the Docker image: `docker build -t speciesnet -f Dockerfile.cpu .`
2. Run the container `docker run -p 8000:8000 speciesnet`
3. The SpeciesNet LitServe server will be running on http://0.0.0.0:8000
4. Test with the following request:

```
curl --location 'http://0.0.0.0:8000/predict' \
--header 'Content-Type: text/plain' \
--data '{
    "instances": [
        {
            "filepath": "test_data/african_elephants.jpg"
        },
        {
            "filepath": "test_data/african_elephants_bw.jpg"
        },
        {
            "filepath": "test_data/african_elephants_cmyk.jpg"
        }
    ]
}
'
```