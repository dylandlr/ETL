from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel


class ExtractorConfig(BaseModel):
    """Base configuration for extractors."""
    name: str
    source_type: str


class BaseExtractor(ABC):
    """Base class for all data extractors.
    
    This class defines the interface that all extractors must implement.
    It provides common functionality such as logging and configuration validation.
    """

    def __init__(self, config: ExtractorConfig):
        """Initialize the extractor with configuration.
        
        Args:
            config: Configuration for the extractor
        """
        self.config = config
        self.logger = structlog.get_logger(name=f"extractor.{config.name}")
        self.logger.info("initializing_extractor", 
                        extractor_name=config.name,
                        source_type=config.source_type)

    @abstractmethod
    async def extract(self, **kwargs: Any) -> Any:
        """Extract data from the source.
        
        Args:
            **kwargs: Additional arguments needed for extraction
            
        Returns:
            Extracted data in the format specific to the extractor
            
        Raises:
            ExtractorError: If extraction fails
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate the connection to the data source.
        
        Returns:
            bool: True if connection is valid, False otherwise
        """
        pass

    async def __aenter__(self) -> 'BaseExtractor':
        """Async context manager entry.
        
        Returns:
            self: The extractor instance
        """
        self.logger.debug("entering_extractor_context")
        return self

    async def __aexit__(self, exc_type: Optional[type], 
                       exc_val: Optional[Exception], 
                       exc_tb: Optional[Any]) -> None:
        """Async context manager exit.
        
        Args:
            exc_type: The type of the exception that was raised
            exc_val: The instance of the exception that was raised
            exc_tb: The traceback of the exception that was raised
        """
        if exc_type is not None:
            self.logger.error("extractor_error",
                            error_type=exc_type.__name__,
                            error_message=str(exc_val))
        self.logger.debug("exiting_extractor_context")