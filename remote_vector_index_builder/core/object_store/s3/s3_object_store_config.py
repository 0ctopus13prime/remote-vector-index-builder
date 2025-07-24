from typing import Optional

from pydantic import BaseModel


class S3ClientConfig(BaseModel):
    """
    Configuration class for creating S3 Boto3 client

    Initialize the S3ObjectStore boto3 client with the following parameters:

    Attributes:
        region_name (str) (required): AWS Region name
        endpoint_url (Optional[str]): Custom S3 endpoint URL
        max_retries (int) (default: 3): Maximum number of retry attempts for failed requests

        AWS Credentials (all optional):
        aws_access_key_id (Optional[str]): AWS Access Key ID
        aws_secret_access_key (Optional[str]): AWS Secret Access Key
        aws_session_token (Optional[str]): Temporary session token for STS credentials

    Note:
        AWS credentials are optional as boto3 will attempt to find credentials:
        For more details see boto3 client documentation:
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    region_name: str
    endpoint_url: Optional[str] = None
    max_retries: int = 3

    # AWS Credentials parameters
    # Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    def __hash__(self):
        """
        Generate a hash value for this configuration.

        Required for @functools.cache to use this class as a dictionary key.
        All attributes that affect the client configuration must be included
        in the hash tuple to ensure proper cache behavior.

        Returns:
            int: Hash value based on all configuration attributes
        """
        return hash(
            (
                self.region_name,
                self.max_retries,
                self.endpoint_url,
                self.aws_access_key_id,
                self.aws_secret_access_key,
                self.aws_session_token,
            )
        )

    def __eq__(self, other):
        """
        Compare this configuration with another for equality.

        Required for @functools.cache to properly identify cache hits.
        Two configurations are equal if all their attributes are equal.

        Args:
            other: Another object to compare with this configuration

        Returns:
            bool: True if other is an S3ClientConfig with identical attributes,
                  False otherwise

        Note:
            Returns NotImplemented for non-S3ClientConfig objects to allow
            Python to try other comparison methods.
        """
        if not isinstance(other, S3ClientConfig):
            return NotImplemented
        return (
            self.region_name == other.region_name
            and self.max_retries == other.max_retries
            and self.endpoint_url == other.endpoint_url
            and self.aws_access_key_id == other.aws_access_key_id
            and self.aws_secret_access_key == other.aws_secret_access_key
            and self.aws_session_token == other.aws_session_token
        )
