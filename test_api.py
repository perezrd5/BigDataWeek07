"""Validate the deployed prediction API against a saved Iris model."""

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from run_pipeline import http_predict, validate_api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/iris.json"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile")
    args = parser.parse_args()
    try:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        result = validate_api(
            lambda body: http_predict(session, args.url, body), args.model.read_bytes(),
        )
        print(json.dumps(result, indent=2))
    except (BotoCoreError, ClientError, OSError, ValueError, RuntimeError) as error:
        print(f"API validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
