import os
import pytest
from datetime import datetime, timedelta
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.monitor import MonitorManagementClient

class TestMonitoring:
    @pytest.fixture(scope="class")
    def logs_client(self):
        """Create Log Analytics client"""
        try:
            credential = DefaultAzureCredential()
            return LogsQueryClient(credential)
        except Exception as e:
            pytest.skip(f"Failed to create Log Analytics client: {str(e)}")

    @pytest.fixture(scope="class")
    def monitor_client(self):
        """Create Monitor Management client"""
        try:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            credential = DefaultAzureCredential()
            return MonitorManagementClient(credential, subscription_id)
        except Exception as e:
            pytest.skip(f"Failed to create Monitor client: {str(e)}")

    def test_alert_rules_configuration(self, monitor_client):
        """Test alert rules are properly configured"""
        resource_group = os.getenv("RESOURCE_GROUP_NAME")
        
        # Get all alert rules
        alert_rules = monitor_client.alert_rules.list_by_resource_group(resource_group)
        
        required_alerts = [
            "HighFailureRate",
            "DataFreshness",
            "ProcessingDelay",
            "ErrorCount"
        ]
        
        configured_alerts = [rule.name for rule in alert_rules]
        missing_alerts = [alert for alert in required_alerts if alert not in configured_alerts]
        
        assert not missing_alerts, f"Missing required alert rules: {missing_alerts}"

    def test_log_analytics_data_collection(self, logs_client):
        """Test Log Analytics is collecting data"""
        workspace_id = os.getenv("LOG_ANALYTICS_WORKSPACE_ID")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        # Query for recent logs
        query = """
        AzureDiagnostics
        | where TimeGenerated > ago(1h)
        | summarize count() by Category
        """
        
        response = logs_client.query_workspace(
            workspace_id=workspace_id,
            query=query,
            timespan=(start_time, end_time)
        )
        
        assert len(response.tables) > 0, "No logs found in Log Analytics workspace"
        assert response.tables[0].row_count > 0, "No log entries in the last hour"

    def test_metric_collection(self, monitor_client):
        """Test metric collection for key resources"""
        resource_group = os.getenv("RESOURCE_GROUP_NAME")
        data_factory_name = os.getenv("DATA_FACTORY_NAME")
        
        # Get Data Factory resource ID
        resource_id = f"/subscriptions/{os.getenv('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{resource_group}/providers/Microsoft.DataFactory/factories/{data_factory_name}"
        
        # Test metric availability
        required_metrics = [
            "PipelineSucceededRuns",
            "PipelineFailedRuns",
            "ActivitySucceededRuns",
            "ActivityFailedRuns"
        ]
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        for metric in required_metrics:
            response = monitor_client.metrics.list(
                resource_id,
                timespan=f"{start_time}/{end_time}",
                interval='PT1H',
                metricnames=metric,
                aggregation='Total'
            )
            
            assert len(response.value) > 0, f"Metric {metric} not found"
            assert len(response.value[0].timeseries) > 0, f"No data points for metric {metric}"

    def test_diagnostic_settings(self, monitor_client):
        """Test diagnostic settings configuration"""
        resource_group = os.getenv("RESOURCE_GROUP_NAME")
        data_factory_name = os.getenv("DATA_FACTORY_NAME")
        
        # Get Data Factory resource ID
        resource_id = f"/subscriptions/{os.getenv('AZURE_SUBSCRIPTION_ID')}/resourceGroups/{resource_group}/providers/Microsoft.DataFactory/factories/{data_factory_name}"
        
        # Get diagnostic settings
        settings = monitor_client.diagnostic_settings.list(resource_id)
        
        has_log_analytics = False
        required_logs = {
            "PipelineRuns": False,
            "TriggerRuns": False,
            "ActivityRuns": False
        }
        
        for setting in settings:
            if setting.workspace_id:  # Check if Log Analytics is configured
                has_log_analytics = True
                for log in setting.logs:
                    if log.category in required_logs and log.enabled:
                        required_logs[log.category] = True
        
        assert has_log_analytics, "Log Analytics workspace not configured in diagnostic settings"
        
        missing_logs = [log for log, enabled in required_logs.items() if not enabled]
        assert not missing_logs, f"Missing required log categories in diagnostic settings: {missing_logs}"

    def test_monitoring_coverage(self, logs_client):
        """Test monitoring coverage across components"""
        workspace_id = os.getenv("LOG_ANALYTICS_WORKSPACE_ID")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        # Query for component coverage
        query = """
        let components = dynamic(['DataFactory', 'Databricks', 'KeyVault', 'Storage']);
        AzureDiagnostics
        | where TimeGenerated > ago(24h)
        | extend Component = case(
            ResourceType contains 'MICROSOFT.DATAFACTORY', 'DataFactory',
            ResourceType contains 'MICROSOFT.DATABRICKS', 'Databricks',
            ResourceType contains 'MICROSOFT.KEYVAULT', 'KeyVault',
            ResourceType contains 'MICROSOFT.STORAGE', 'Storage',
            'Other')
        | where Component in (components)
        | summarize LastLog = max(TimeGenerated) by Component
        | extend TimeSinceLastLog = now() - LastLog
        | project Component, TimeSinceLastLog
        """
        
        response = logs_client.query_workspace(
            workspace_id=workspace_id,
            query=query,
            timespan=(start_time, end_time)
        )
        
        if len(response.tables) > 0 and response.tables[0].row_count > 0:
            for row in response.tables[0].rows:
                component, time_since_last = row
                time_since_last = float(time_since_last)  # Convert to hours
                assert time_since_last <= 2, f"No recent logs from {component} in the last 2 hours"
        else:
            pytest.fail("No monitoring data found for key components")
