param(
    [string]$Stack = "iris-pipeline",
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

function Invoke-Aws {
    & aws --region $Region --no-cli-pager @args
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI command failed (exit code $LASTEXITCODE). See the service error above."
    }
}

Push-Location $PSScriptRoot
try {
    $identity = Invoke-Aws sts get-caller-identity --output json | ConvertFrom-Json
    $assetBucket = "iris-deploy-$($identity.Account)-$Region"
    $buckets = Invoke-Aws s3api list-buckets --query "Buckets[].Name" --output json |
        ConvertFrom-Json
    if ($assetBucket -notin $buckets) {
        $create = @("s3api", "create-bucket", "--bucket", $assetBucket)
        if ($Region -ne "us-east-1") {
            $create += @("--create-bucket-configuration", "LocationConstraint=$Region")
        }
        Invoke-Aws @create | Out-Null
    }
    Invoke-Aws s3api put-public-access-block --bucket $assetBucket --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

    New-Item -ItemType Directory -Path .aws-sam -Force | Out-Null
    $packagedTemplate = ".aws-sam/packaged.yaml"
    Invoke-Aws cloudformation package --template-file template.yaml --s3-bucket $assetBucket --s3-prefix $Stack --output-template-file $packagedTemplate
    Invoke-Aws cloudformation validate-template --template-body "file://$packagedTemplate" |
        Out-Null
    Invoke-Aws cloudformation deploy --template-file $packagedTemplate --stack-name $Stack --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset
    Invoke-Aws cloudformation describe-stacks --stack-name $Stack --query "Stacks[0].Outputs" --output table
}
finally {
    Pop-Location
}
