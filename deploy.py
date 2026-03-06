# WARNING: This snippet is not yet compatible with SageMaker version >= 3.0.0.
# To use this snippet, install a compatible version:
# pip install 'sagemaker<3.0.0'
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'meta-llama/Llama-3.3-70B-Instruct',
	'SM_NUM_GPUS': json.dumps(8),
	'HF_TOKEN': '<REPLACE WITH YOUR TOKEN>'
}

assert hub['HF_TOKEN'] != '<REPLACE WITH YOUR TOKEN>', "You have to provide a token."

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	image_uri=get_huggingface_llm_image_uri("huggingface",version="3.3.6"),
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g6.48xlarge",
	container_startup_health_check_timeout=2400,
  )
  
# send request
predictor.predict({
	"inputs": "Hi, what can you help me with?",
})
