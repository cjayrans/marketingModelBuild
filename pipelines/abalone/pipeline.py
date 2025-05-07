"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )

    # Processing step for feature engineering

    # Define containerized environment: Encapsulates the environment definition for the broader preprocessing step
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    # Configure job I/O and script: Defines how the SKLearnProcessor will execute its job - declaring exact inputs/outputs and parameters for this step
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )
    # Register step in pipeline graph: Wraps the preprocessing run configuration (step_args) into a named pipeline step that can be chained to other steps.
    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        step_args=step_args,
    )

    # TRAINING STEP FOR GENERATING MODEL ARTIFACTS
    # S3 destination for trained model
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    
    # Select XGBoost training environment: Retrieves the Docker image URI for the SageMaker-provided XGBoost container
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    # Defines the training job configuration using SageMaker’s Estimator
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    # Sets the model's training hyperparameters (specific to XGBoost)
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    # Execute training using preprocessed data
    # Runs the training job by specifying the S3 inputs created by the preprocessing step (step_process)
    # This is a dependency chain: the training step now officially depends on the outputs of the preprocessing step.
    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput( # Wraps that path (S3 URI note below) as input to SageMaker's built-in training API
                # Dynamically grabs the output S3 URI from the preprocessing step
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    # Register in pipeline: Wraps the fit(...) call into a named SageMaker TrainingStep to be included in the pipeline DAG
    step_train = TrainingStep(
        name="TrainAbaloneModel", # Clearly named steps show up in SageMaker Studio's visual pipeline graph
        step_args=step_args,
    )

    # PROCESSING STEP FOR EVALUATION
    # This step runs the evaluate.py script inside a Python container and stores its outputs for use in a conditional logic step
    
    # Defines the environment for running a custom Python evaluation script
    script_eval = ScriptProcessor( # runs arbitrary Python scripts inside containers
        image_uri=image_uri, # reuses the XGBoost container (which includes Python + dependencies)
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    # Tells the ScriptProcessor how to run the evaluation job
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts, # Pulled from the training step’s output (model.tar.gz)
                destination="/opt/ml/processing/model", # Placed into /opt/ml/processing/model inside the container
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[ 
                    "test"
                ].S3Output.S3Uri, # Comes from the preprocessing step
                destination="/opt/ml/processing/test", # Placed into /opt/ml/processing/test
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"), # Evaluation results (MSE + stddev) written to /opt/ml/processing/evaluation
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"), # The evaluate.py script writes the resulting JSON output containing evaluation results 
    )
    # Registers the location of the evaluation output so it can be referenced by name and path in later steps
    evaluation_report = PropertyFile(
        name="AbaloneEvaluationReport",
        output_name="evaluation", # use of "evaluation" must match the name in ProcessingOutput(...)
        path="evaluation.json", # is the exact file written by evaluate.py containing evaluation metric results - This allows downstream steps (like ConditionStep) to extract the MSE value with JsonGet(...)
    )
    # Wraps the evaluation logic into a named pipeline step
    # Stores evaluation.json as part of the pipeline’s metadata
    # Makes outputs available to ConditionStep logic
    step_eval = ProcessingStep(
        name="EvaluateAbaloneModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # REGISTER MODEL STEP THAT WILL BE CONDITIONALLY EXECUTED - CONTAINS TWO SUB-SECTIONS
    # 1. Model registration logic (defining how the model is registered if approved)
    # 2. A conditional branch that checks whether the model meets quality requirements before allowing registration

    # Defines model evaluation metrics (in this case, MSE) to be associated with the model in the SageMaker Model Registry
    model_metrics = ModelMetrics( # ModelMetrics object → Attached to the model package when it’s registered
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] # step_eval.arguments["..."] → Dynamically resolves the path to evaluation.json
            ),
            content_type="application/json"
        )
    )

    # Defines a SageMaker Model object using:
        # The training step’s model artifact (model.tar.gz)
        # The same container image used during training (image_uri)
    # This creates a deployable MODEL OBJECT that can be used for:
        # Real-time inference (SageMaker Endpoint)
        # Batch transform
        # Model registration
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    # Registers the model to a Model Package Group in the SageMaker Model Registry
    # Enables governance, versioning, and deployment approval workflows
    # Central place to store models before deployment
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name, # Groups versions of the same model lineage (e.g., "AbalonePackageGroup")
        approval_status=model_approval_status,
        model_metrics=model_metrics, # Injects evaluation results to be stored as meta data in model registry
    )

    # Creates a pipeline step to register the model — but this step won’t run automatically. It will be conditionally triggered in the next block
    step_register = ModelStep(
        name="RegisterAbaloneModel",
        step_args=step_args,
    )

    # Condition step for evaluating model quality and branching execution
    # Checks the MSE value from evaluation.json. If mse <= 6.0, the condition is met
    # This is your model quality gate — only models that perform well get registered and become candidates for deployment.
    cond_lte = ConditionLessThanOrEqualTo( # Allows branching based on this metric
        left=JsonGet( # pulls the MSE value out of the evaluation_report
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )

     # Creates a conditional pipeline branch:
        # If the MSE condition passes → run step_register
        # If not → do nothing (or you could add alerts, logging, etc.)
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # PIPELINE INSTANCE
    # Ties together everything into a unified pipeline definition that you can create, update, or start programmatically
    pipeline = Pipeline(
        name=pipeline_name, # The name you will see in the SageMaker UI (e.g., “AbalonePipeline”)
        parameters=[ # These are runtime pipeline parameters defined earlier, allowing flexibility without changing the code.
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond], # This defines the directed acyclic graph (DAG) for the pipeline
        sagemaker_session=pipeline_session, # Connects this pipeline definition to the SageMaker session that knows how to run it in your AWS account.
    )
    return pipeline
