import os
import json
import pytest
from dotenv import load_dotenv
from pathlib import Path

def load_terraform_outputs(terraform_dir):
    """Load Terraform outputs from the state file"""
    try:
        # Run terraform output command
        import subprocess
        result = subprocess.run(
            ['terraform', 'output', '-json'],
            cwd=terraform_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Failed to get terraform outputs: {result.stderr}")
        return json.loads(result.stdout)
    except Exception as e:
        pytest.skip(f"Terraform outputs not available: {str(e)}")

def get_required_env_vars():
    """List of required environment variables"""
    return [
        # Azure Resource Information
        "RESOURCE_GROUP_NAME",
        "KEY_VAULT_URI",
        "STORAGE_ACCOUNT_NAME",
        "DATALAKE_URL",
        
        # Data Factory Information
        "DATA_FACTORY_NAME",
        "DATA_FACTORY_ID",
        
        # Databricks Information
        "DATABRICKS_HOST",
        "DATABRICKS_WORKSPACE_ID",
        "DATABRICKS_WORKSPACE_RESOURCE_ID",
        
        # SQL Database Information
        "SQL_SERVER_ID",
        "SQL_DATABASE_ID",
        
        # Environment Information
        "ENVIRONMENT",
        "LOCATION"
    ]

class TestEnvironmentConfiguration:
    @pytest.fixture(scope="class")
    def env_file_path(self):
        """Get the .env file path"""
        root_dir = Path(__file__).parent.parent.parent
        return root_dir / ".env"

    @pytest.fixture(scope="class")
    def terraform_dir(self):
        """Get the Terraform directory path"""
        root_dir = Path(__file__).parent.parent.parent
        return root_dir / "infrastructure"

    def test_env_file_exists(self, env_file_path):
        """Test that .env file exists"""
        assert env_file_path.exists(), f".env file not found at {env_file_path}"

    def test_env_file_readable(self, env_file_path):
        """Test that .env file is readable"""
        assert os.access(env_file_path, os.R_OK), f".env file at {env_file_path} is not readable"

    def test_required_variables_present(self, env_file_path):
        """Test that all required variables are present in .env"""
        load_dotenv(env_file_path)
        missing_vars = []
        
        for var in get_required_env_vars():
            if not os.getenv(var):
                missing_vars.append(var)
        
        assert not missing_vars, f"Missing required environment variables: {', '.join(missing_vars)}"

    def test_variable_values_not_empty(self, env_file_path):
        """Test that environment variables have non-empty values"""
        load_dotenv(env_file_path)
        empty_vars = []
        
        for var in get_required_env_vars():
            value = os.getenv(var)
            if value is not None and not str(value).strip():
                empty_vars.append(var)
        
        assert not empty_vars, f"Environment variables with empty values: {', '.join(empty_vars)}"

    def test_terraform_output_matches_env(self, env_file_path, terraform_dir):
        """Test that Terraform outputs match environment variables"""
        load_dotenv(env_file_path)
        terraform_outputs = load_terraform_outputs(terraform_dir)
        
        # Map of environment variables to their corresponding Terraform output keys
        env_to_terraform_map = {
            "RESOURCE_GROUP_NAME": "resource_group_name",
            "KEY_VAULT_URI": "key_vault_uri",
            "STORAGE_ACCOUNT_NAME": "storage_account_name",
            "DATALAKE_URL": "datalake_url",
            "DATA_FACTORY_NAME": "data_factory_name",
            "DATA_FACTORY_ID": "data_factory_id",
            "SQL_SERVER_ID": "sql_server_id",
            "SQL_DATABASE_ID": "sql_database_id"
        }
        
        mismatched_vars = []
        for env_var, tf_output in env_to_terraform_map.items():
            env_value = os.getenv(env_var)
            if tf_output in terraform_outputs:
                tf_value = terraform_outputs[tf_output]["value"]
                if env_value != tf_value:
                    mismatched_vars.append(f"{env_var}: env='{env_value}' != terraform='{tf_value}'")
        
        assert not mismatched_vars, f"Mismatched values between .env and Terraform outputs:\n" + "\n".join(mismatched_vars)

    def test_databricks_connection(self, env_file_path):
        """Test Databricks connection using environment variables"""
        load_dotenv(env_file_path)
        
        required_databricks_vars = [
            "DATABRICKS_HOST",
            "DATABRICKS_TOKEN"
        ]
        
        missing_vars = []
        for var in required_databricks_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Skipping Databricks connection test. Missing variables: {', '.join(missing_vars)}")
        
        try:
            from databricks_cli.sdk.api_client import ApiClient
            from databricks_cli.clusters.api import ClusterApi
            
            api_client = ApiClient(
                host=os.getenv("DATABRICKS_HOST"),
                token=os.getenv("DATABRICKS_TOKEN")
            )
            
            # Try to list clusters to verify connection
            clusters_api = ClusterApi(api_client)
            clusters_api.list_clusters()
            
        except Exception as e:
            pytest.fail(f"Failed to connect to Databricks: {str(e)}")

    def test_key_vault_connection(self, env_file_path):
        """Test Azure Key Vault connection using environment variables"""
        load_dotenv(env_file_path)
        
        key_vault_uri = os.getenv("KEY_VAULT_URI")
        if not key_vault_uri:
            pytest.skip("Skipping Key Vault connection test. KEY_VAULT_URI not set")
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            credential = DefaultAzureCredential()
            secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
            
            # Try to list secrets to verify connection
            list(secret_client.list_properties_of_secrets(max_page_size=1))
            
        except Exception as e:
            pytest.fail(f"Failed to connect to Key Vault: {str(e)}")
