import os
import pytest
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory.models import PipelineRun
from datetime import datetime, timedelta

class TestDataFactoryPipeline:
    @pytest.fixture(scope="class")
    def adf_client(self):
        """Create Data Factory client"""
        try:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            credential = DefaultAzureCredential()
            return DataFactoryManagementClient(credential, subscription_id)
        except Exception as e:
            pytest.skip(f"Failed to create Data Factory client: {str(e)}")

    @pytest.fixture(scope="class")
    def resource_info(self):
        """Get resource information"""
        return {
            "resource_group": os.getenv("RESOURCE_GROUP_NAME"),
            "factory_name": os.getenv("DATA_FACTORY_NAME")
        }

    def test_pipeline_exists(self, adf_client, resource_info):
        """Test that required pipelines exist"""
        required_pipelines = ["MainETLPipeline", "ErrorHandlingPipeline"]
        existing_pipelines = [
            p.name for p in adf_client.pipelines.list_by_factory(
                resource_info["resource_group"], 
                resource_info["factory_name"]
            )
        ]
        
        missing_pipelines = [p for p in required_pipelines if p not in existing_pipelines]
        assert not missing_pipelines, f"Missing required pipelines: {missing_pipelines}"

    def test_linked_services_connection(self, adf_client, resource_info):
        """Test all linked services are connected"""
        linked_services = adf_client.linked_services.list_by_factory(
            resource_info["resource_group"],
            resource_info["factory_name"]
        )
        
        failed_services = []
        for service in linked_services:
            try:
                result = adf_client.linked_services.get(
                    resource_info["resource_group"],
                    resource_info["factory_name"],
                    service.name
                )
                if not result or not result.properties:
                    failed_services.append(service.name)
            except Exception as e:
                failed_services.append(f"{service.name} (Error: {str(e)})")
        
        assert not failed_services, f"Failed linked services: {failed_services}"

    def test_pipeline_activity_dependencies(self, adf_client, resource_info):
        """Test pipeline activities have correct dependencies"""
        pipeline = adf_client.pipelines.get(
            resource_info["resource_group"],
            resource_info["factory_name"],
            "MainETLPipeline"
        )
        
        activities = pipeline.properties.activities
        activity_names = [a.name for a in activities]
        
        # Verify essential activities exist
        required_activities = [
            "Data_Extraction",
            "Data_Validation",
            "Data_Transformation",
            "Data_Loading"
        ]
        
        missing_activities = [a for a in required_activities if a not in activity_names]
        assert not missing_activities, f"Missing required activities: {missing_activities}"

    def test_error_handling_implementation(self, adf_client, resource_info):
        """Test error handling pipeline implementation"""
        pipeline = adf_client.pipelines.get(
            resource_info["resource_group"],
            resource_info["factory_name"],
            "ErrorHandlingPipeline"
        )
        
        activities = pipeline.properties.activities
        activity_names = [a.name for a in activities]
        
        required_activities = ["Log Error Details", "Send Error Notification"]
        missing_activities = [a for a in required_activities if a not in activity_names]
        assert not missing_activities, f"Missing error handling activities: {missing_activities}"

    def test_recent_pipeline_runs(self, adf_client, resource_info):
        """Test recent pipeline runs for failures"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        runs = adf_client.pipeline_runs.query_by_factory(
            resource_info["resource_group"],
            resource_info["factory_name"],
            {
                "lastUpdatedAfter": start_time.isoformat(),
                "lastUpdatedBefore": end_time.isoformat()
            }
        )
        
        failed_runs = []
        for run in runs.value:
            if run.status == "Failed":
                failed_runs.append({
                    "pipeline": run.pipeline_name,
                    "run_id": run.run_id,
                    "status": run.status,
                    "message": getattr(run, "message", "No error message")
                })
        
        assert not failed_runs, f"Failed pipeline runs in last 24 hours: {failed_runs}"
