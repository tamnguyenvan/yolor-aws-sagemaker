import time
import json
import os
import io
import argparse
import shutil
import tarfile

import boto3
import botocore
import numpy as np
import sagemaker
from PIL import Image

from inference import input_fn, model_fn, predict_fn, output_fn


def fetch_model(model_data):
    """Untar the model.tar.gz object either from local file system
    or a S3 location
    Args:
        model_data (str): either a path to local file system starts with
        file:/// that points to the `model.tar.gz` file or an S3 link
        starts with s3:// that points to the `model.tar.gz` file
    Returns:
        model_dir (str): the directory that contains the uncompress model
        checkpoint files
    """

    model_dir = "/tmp/model"
    model_path = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(model_path):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # if model_data.startswith("file"):
        #     _check_model(model_data)
        #     shutil.copy2(
        #         os.path.join(model_dir, "model.tar.gz"), os.path.join(model_dir, "model.tar.gz")
        #     )
        elif model_data.startswith("s3"):
            # get bucket name and object key
            bucket_name = model_data.split("/")[2]
            key = "/".join(model_data.split("/")[3:])

            s3 = boto3.resource("s3")
            try:
                s3.Bucket(bucket_name).download_file(key, os.path.join(model_dir, "model.tar.gz"))
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    print("the object does not exist.")
                else:
                    raise

    # untar the model
    tar = tarfile.open(os.path.join(model_dir, "model.tar.gz"))
    tar.extractall(model_dir)
    tar.close()

    return model_dir


def test(model_data):
    # decompress the model.tar.gz file
    print('Fetching model')
    model_dir = fetch_model(model_data)

    # load the model
    print('Loading model')
    net = model_fn(model_dir)

    # simulate some input data to test transform_fn
    x = np.array(Image.open('test.jpg').convert('RGB'))

    t0 = time.time()

    # "send" the bin_stream to the endpoint for inference
    # inference container calls transform_fn to make an inference
    # and get the response body for the caller
    
    print('Predicting')
    content_type = "image/jpeg"
    buffer = io.BytesIO()
    img = Image.fromarray(x)
    img.save(buffer, format='jpeg')
    input_object = input_fn(buffer.getvalue(), content_type)
    print(input_object.shape)
    predictions = predict_fn(input_object, net)
    res = output_fn(predictions, 'application/json')
    print(res)
    t1 = time.time()
    print('Duration:', t1 - t0)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Path to model on s3')
    args = parser.parse_args()
    model_data = args.model
    print(model_data)
    test(model_data)