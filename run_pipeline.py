"""Run and verify the Iris pipeline locally or against its deployed AWS stack."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import boto3
import numpy as np
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline import (
    FEATURES, LABELS, clean_csv, inference, make_csv, predict, read_s3, transform,
)

MODEL_KEY = "models/iris.json"
EXAMPLE = dict(zip(FEATURES, [5.1, 3.5, 1.4, 0.2]))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def public_dataset():
    iris = load_iris()
    return make_csv(
        (*measurements, iris.target_names[target])
        for measurements, target in zip(iris.data, iris.target, strict=True)
    )


def train(cleaned):
    rows = list(csv.DictReader(io.StringIO(cleaned.decode("utf-8"))))
    x = [[float(row[name]) for name in FEATURES] for row in rows]
    y = [row["species"] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42,
    )
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    estimator.fit(x_train, y_train)
    scaler, classifier = (step for _, step in estimator.steps)
    model = {
        "format_version": 1,
        "features": FEATURES,
        "classes": classifier.classes_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
    }
    model["model_id"] = hashlib.sha256(encode(model)).hexdigest()
    artifact = encode(model)
    restored = json.loads(artifact)
    expected = estimator.predict(x_test)
    expected_probabilities = estimator.predict_proba(x_test)
    responses = [predict(restored, row) for row in x_test]
    require([item["prediction"] for item in responses] == expected.tolist(),
            "Serialized model predictions differ from scikit-learn")
    actual_probabilities = [
        [item["probabilities"][label] for label in classifier.classes_]
        for item in responses
    ]
    require(np.allclose(actual_probabilities, expected_probabilities, rtol=1e-10, atol=1e-12),
            "Serialized model probabilities differ from scikit-learn")
    metrics = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": accuracy_score(y_test, expected),
        "macro_f1": f1_score(y_test, expected, average="macro"),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, expected, labels=LABELS).tolist(),
    }
    return artifact, metrics


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def wait_for_etl(client, arn, raw_key):
    execution = client.start_execution(
        stateMachineArn=arn, input=json.dumps({"raw_key": raw_key}),
    )["executionArn"]
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        state = client.describe_execution(executionArn=execution)
        status = state["status"]
        if status == "SUCCEEDED":
            return execution, json.loads(state["output"])
        if status not in {"RUNNING", "PENDING_REDRIVE"}:
            raise RuntimeError(f"ETL {status} ({execution}): {state.get('cause', '')}")
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for ETL execution {execution}")


def http_predict(session, url, body):
    headers = {"Content-Type": "application/json"}
    signed = AWSRequest(method="POST", url=url, data=body, headers=headers)
    credentials = session.get_credentials().get_frozen_credentials()
    SigV4Auth(credentials, "execute-api", session.region_name).add_auth(signed)
    request = Request(url, data=body, headers=dict(signed.headers), method="POST")
    try:
        with urlopen(request, timeout=30) as result:
            return result.status, json.load(result)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


def validate_api(send, artifact):
    model = json.loads(artifact)
    status, payload = send(encode(EXAMPLE))
    require(status == 200, f"Expected HTTP 200, got {status}: {payload}")
    expected = predict(model, list(EXAMPLE.values()))
    require(payload["model_id"] == model["model_id"], "API served a different model")
    require(payload["prediction"] == expected["prediction"] == "setosa",
            f"Unexpected class: {payload}")
    probabilities = payload["probabilities"]
    require(set(probabilities) == set(LABELS)
            and math.isclose(sum(probabilities.values()), 1.0)
            and all(math.isclose(probabilities[label], expected["probabilities"][label])
                    for label in LABELS), "Unexpected class probabilities")

    nonfinite = encode({**EXAMPLE, FEATURES[0]: 0}).replace(b': 0,', b': NaN,')
    invalid_bodies = [b"{", b"[]", b"{}", encode({**EXAMPLE, FEATURES[0]: True}),
                      encode({**EXAMPLE, FEATURES[0]: -1}), nonfinite]
    for body in invalid_bodies:
        status, error = send(body)
        require(status == 400 and "error" in error,
                f"Expected HTTP 400 for {body!r}, got {status}: {error}")
    return {"prediction_response": payload, "invalid_requests_rejected": len(invalid_bodies)}


def local_cleaning_check():
    fixture = make_csv([
        (5.1, 3.5, 1.4, 0.2, " SETOSA "),
        (5.1, 3.5, 1.4, 0.2, "setosa"),
        ("nan", 3.5, 1.4, 0.2, "setosa"),
        (5.1, "", 1.4, 0.2, "setosa"),
        (5.1, 3.5, 1.4, 0.2, "unknown"),
    ])
    key = "raw/fixture.csv"
    output_key = "processed/fixture.csv"
    s3 = Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(fixture)}
    with patch("boto3.client", return_value=s3), patch.dict(
        os.environ, {"RAW_BUCKET": "raw-bucket", "PROCESSED_BUCKET": "processed-bucket"},
    ):
        result = transform({"raw_key": key}, None)
    s3.get_object.assert_called_once_with(Bucket="raw-bucket", Key=key)
    s3.put_object.assert_called_once_with(
        Bucket="processed-bucket", Key=output_key,
        Body=make_csv([(5.1, 3.5, 1.4, 0.2, "setosa")]), ContentType="text/csv",
    )
    counts = dict(input_rows=5, invalid_rows=3, duplicate_rows=1, output_rows=1)
    require(result == {"raw_key": key, "processed_key": output_key, **counts},
            f"Unexpected ETL result: {result}")


def run_local(raw):
    local_cleaning_check()
    cleaned, counts = clean_csv(raw)
    artifact, metrics = train(cleaned)
    with patch("src.pipeline.load_model", return_value=json.loads(artifact)):
        def send(body):
            result = inference({"body": body.decode()}, None)
            return result["statusCode"], json.loads(result["body"])

        checks = validate_api(send, artifact)
    return artifact, {"mode": "local", "etl": counts, "metrics": metrics, "api": checks}


def run_cloud(raw, stack_name, region, profile):
    session = boto3.Session(profile_name=profile, region_name=region)
    stack = session.client("cloudformation").describe_stacks(
        StackName=stack_name,
    )["Stacks"][0]
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack["Outputs"]}
    s3 = session.client("s3")
    raw_key = f"raw/iris-{uuid4().hex}.csv"
    s3.upload_fileobj(io.BytesIO(raw), outputs["RawBucketName"], raw_key,
                      ExtraArgs={"ContentType": "text/csv"})
    execution, etl = wait_for_etl(session.client("stepfunctions"), outputs["WorkflowArn"], raw_key)
    bucket = outputs["ProcessedBucketName"]
    require(etl["raw_key"] == raw_key and
            etl["processed_key"] == raw_key.replace("raw/", "processed/", 1),
            "ETL returned unexpected object keys")
    cleaned = read_s3(s3, bucket, etl["processed_key"])
    expected, counts = clean_csv(raw)
    require(cleaned == expected and all(etl[key] == value for key, value in counts.items()),
            "S3 output or ETL counts differ from local cleaning")
    artifact, metrics = train(cleaned)
    s3.upload_fileobj(io.BytesIO(artifact), bucket, MODEL_KEY,
                      ExtraArgs={"ContentType": "application/json"})
    require(read_s3(s3, bucket, MODEL_KEY) == artifact, "S3 model failed round-trip verification")
    url = outputs["PredictionUrl"]
    api = validate_api(lambda body: http_predict(session, url, body), artifact)
    return artifact, {
        "mode": "aws", "execution_arn": execution, "api_url": url,
        "etl": etl, "metrics": metrics, "api": api,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", help="Run against this CloudFormation stack")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        raw = public_dataset()
        if args.stack:
            artifact, report = run_cloud(raw, args.stack, args.region, args.profile)
        else:
            artifact, report = run_local(raw)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "iris.json").write_bytes(artifact)
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
    except (BotoCoreError, ClientError, OSError, ValueError, RuntimeError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
