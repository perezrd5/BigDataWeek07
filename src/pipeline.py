"""Shared data contract and Lambda handlers for the Iris pipeline."""

import base64
import binascii
import csv
import io
import json
import logging
import math
import os

FEATURES = ("sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm")
LABELS = ("setosa", "versicolor", "virginica")
COLUMNS = (*FEATURES, "species")


def read_s3(s3, bucket, key):
    with s3.get_object(Bucket=bucket, Key=key)["Body"] as body:
        return body.read()


def make_csv(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def clean_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if reader.fieldnames is None or sorted(reader.fieldnames) != sorted(COLUMNS):
        raise ValueError("Unexpected Iris CSV header")
    kept = {}
    counts = dict(input_rows=0, invalid_rows=0, duplicate_rows=0)
    for row in reader:
        counts["input_rows"] += 1
        try:
            features = tuple(float(row[field]) for field in FEATURES)
            species = row["species"].strip().lower()
            if None in row or species not in LABELS or any(
                not math.isfinite(value) or value <= 0 for value in features
            ):
                raise ValueError("Invalid row")
        except (ValueError, TypeError, AttributeError):
            counts["invalid_rows"] += 1
            continue
        normalized = (*features, species)
        if normalized in kept:
            counts["duplicate_rows"] += 1
        kept[normalized] = None
    if not kept:
        raise ValueError("The dataset contains no usable rows")
    counts["output_rows"] = len(kept)
    return make_csv(kept), counts


def transform(event, context):
    import boto3

    s3 = boto3.client("s3")
    raw_key = event["raw_key"]
    if not raw_key.startswith("raw/") or not raw_key.endswith(".csv"):
        raise ValueError("Expected a CSV key under raw/")
    raw = read_s3(s3, os.environ["RAW_BUCKET"], raw_key)
    cleaned, counts = clean_csv(raw)
    processed_key = "processed/" + raw_key.removeprefix("raw/")
    s3.put_object(
        Bucket=os.environ["PROCESSED_BUCKET"], Key=processed_key,
        Body=cleaned, ContentType="text/csv",
    )
    return {"raw_key": raw_key, "processed_key": processed_key, **counts}


def feature_vector(payload):
    if not isinstance(payload, dict) or set(payload) != set(FEATURES):
        raise ValueError("Provide exactly these fields: " + ", ".join(FEATURES))
    values = []
    for name in FEATURES:
        value = payload[name]
        if type(value) not in (int, float) or not 0 < value <= 100:
            raise ValueError(f"{name} must be a number in (0, 100] cm")
        values.append(value)
    return values


def predict(model, values):
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(values, model["mean"], model["scale"], strict=True)
    ]
    scores = [
        bias + sum(weight * value for weight, value in zip(weights, standardized, strict=True))
        for weights, bias in zip(model["coef"], model["intercept"], strict=True)
    ]
    largest = max(scores)
    exponentials = [math.exp(score - largest) for score in scores]
    total = sum(exponentials)
    probabilities = dict(zip(
        model["classes"], (value / total for value in exponentials), strict=True
    ))
    return {"prediction": max(probabilities, key=probabilities.get),
            "probabilities": probabilities, "model_id": model["model_id"]}


def response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, allow_nan=False)}


def load_model():
    import boto3

    return json.loads(read_s3(boto3.client("s3"), os.environ["MODEL_BUCKET"],
                              os.environ["MODEL_KEY"]))


def inference(event, context):
    try:
        body = event.get("body")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body, validate=True)
        values = feature_vector(json.loads(body))
    except (TypeError, ValueError, UnicodeError, binascii.Error) as error:
        return response(400, {"error": str(error)})
    try:
        return response(200, predict(load_model(), values))
    except Exception:
        logging.exception("Failed to load or evaluate the Iris model")
        return response(503, {"error": "Model unavailable"})
