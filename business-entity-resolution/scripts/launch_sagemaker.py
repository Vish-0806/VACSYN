import boto3

REGION = "ap-south-1"
ROLE_ARN = "arn:aws:iam::962565294311:role/SageMakerEntityResolutionRole"
BUCKET = "amazon-ml-2026-vishal-entity-resolution"

JOB_NAME = "entity-resolution-lightgbm-20260927"

sm = boto3.client("sagemaker", region_name=REGION)

response = sm.create_training_job(
    TrainingJobName=JOB_NAME,
    RoleArn=ROLE_ARN,
    AlgorithmSpecification={
        "TrainingImage": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/sklearn:1.2-1-cpu-py3",
        "TrainingInputMode": "File",
    },
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/train/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/tab-separated-values",
            "InputMode": "File",
        }
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{BUCKET}/sagemaker-output/"
    },
    ResourceConfig={
        "InstanceType": "ml.r5.4xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 200,
    },
    StoppingCondition={
        "MaxRuntimeInSeconds": 86400
    },
    HyperParameters={
        "max-candidates-per-s1": "50",
        "val-ratio": "0.2",
        "random-state": "42",
        "n-estimators": "500",
        "learning-rate": "0.05",
        "early-stopping-rounds": "50",
    },
    Environment={
        "SAGEMAKER_PROGRAM": "scripts/sagemaker_train.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": (
            f"s3://{BUCKET}/code/business-entity-resolution-sagemaker.tar.gz"
        ),
        "SAGEMAKER_REQUIREMENTS": "requirements.txt",
    },
)

print("TRAINING JOB SUBMITTED")
print("Job name:", JOB_NAME)
print("ARN:", response["TrainingJobArn"])