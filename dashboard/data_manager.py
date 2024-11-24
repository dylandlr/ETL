from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import os
from datetime import datetime
import streamlit as st
from pandas_profiling import ProfileReport
import json

class DataLakeManager:
    def __init__(self, account_name="etldatalakedev"):
        self.account_url = f"https://{account_name}.dfs.core.windows.net"
        self.credential = DefaultAzureCredential()
        self.service_client = DataLakeServiceClient(
            account_url=self.account_url,
            credential=self.credential
        )

    def list_containers(self):
        """List all containers in the data lake"""
        return [container.name for container in self.service_client.list_file_systems()]

    def list_files(self, container_name, path=""):
        """List files in a specific container and path"""
        try:
            file_system_client = self.service_client.get_file_system_client(container_name)
            paths = file_system_client.get_paths(path=path)
            return [{"name": path.name, "size": path.content_length} for path in paths]
        except Exception as e:
            st.error(f"Error listing files: {str(e)}")
            return []

    def upload_file(self, container_name, local_file, remote_path):
        """Upload a file to the data lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container_name)
            file_client = file_system_client.get_file_client(remote_path)
            
            with open(local_file, "rb") as data:
                file_client.upload_data(data, overwrite=True)
            
            return True, "File uploaded successfully"
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"

    def download_file(self, container_name, remote_path, local_path):
        """Download a file from the data lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container_name)
            file_client = file_system_client.get_file_client(remote_path)
            
            with open(local_path, "wb") as local_file:
                download = file_client.download_file()
                local_file.write(download.readall())
            
            return True, "File downloaded successfully"
        except Exception as e:
            return False, f"Error downloading file: {str(e)}"

    def read_csv_from_datalake(self, container_name, remote_path):
        """Read a CSV file directly from the data lake into a pandas DataFrame"""
        try:
            file_system_client = self.service_client.get_file_system_client(container_name)
            file_client = file_system_client.get_file_client(remote_path)
            
            download = file_client.download_file()
            return pd.read_csv(download.readall())
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            return None

    def generate_data_profile(self, df, filename):
        """Generate a data profile report for a DataFrame"""
        try:
            profile = ProfileReport(df, title=f"Data Profile Report - {filename}")
            return profile.to_html()
        except Exception as e:
            st.error(f"Error generating profile: {str(e)}")
            return None

def render_data_management_ui():
    """Render the data management interface in Streamlit"""
    st.header("Data Management")
    
    # Initialize DataLakeManager
    data_manager = DataLakeManager()
    
    # Container selection
    containers = data_manager.list_containers()
    container = st.selectbox("Select Container", containers)
    
    # File upload section
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'json'])
    
    if uploaded_file is not None:
        # Create a timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        remote_path = f"input/{filename}"
        
        # Save uploaded file temporarily
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to Data Lake
        success, message = data_manager.upload_file(container, filename, remote_path)
        
        # Clean up temporary file
        os.remove(filename)
        
        if success:
            st.success(message)
        else:
            st.error(message)
    
    # File listing section
    st.subheader("Available Files")
    files = data_manager.list_files(container, "input/")
    
    if files:
        for file in files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(file["name"])
            
            with col2:
                st.write(f"{file['size']/1024:.2f} KB")
            
            with col3:
                if st.button(f"Analyze {os.path.basename(file['name'])}"):
                    df = data_manager.read_csv_from_datalake(container, file["name"])
                    if df is not None:
                        st.write("Data Preview:")
                        st.dataframe(df.head())
                        
                        st.write("Data Profile:")
                        profile_html = data_manager.generate_data_profile(df, os.path.basename(file["name"]))
                        if profile_html:
                            st.components.v1.html(profile_html, height=600, scrolling=True)
    else:
        st.info("No files found in the selected container")
