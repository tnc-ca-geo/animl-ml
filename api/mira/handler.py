import json
from log_cfg import logger


def classify(event, context):
    print("MiraClassify firing")
    logger.debug('event: {}'.format(event))
    logger.debug('event: {}'.format(context))
    if event['body']:
        data = json.loads(event['body'])
        logger.debut("data: ", data)
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
