"""
Object storage configuration for StudySense using Supabase Storage
Handles static KB assets, anonymized exports, and file storage
"""

import logging
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.supabase_service import supabase_service

logger = logging.getLogger(__name__)

class StudySenseStorage:
    """StudySense storage operations using Supabase Storage"""

    def __init__(self):
        self.supabase_service = supabase_service
        self.bucket_name = settings.STORAGE_BUCKET_NAME

    def ensure_bucket_exists(self) -> bool:
        """Ensure the storage bucket exists"""
        try:
            # Try to list files to check if bucket is accessible
            self.supabase_service.list_files(self.bucket_name)
            return True
        except:
            # If bucket doesn't exist or is not accessible, try to create it
            try:
                result = self.supabase_service.create_bucket(
                    bucket_name=self.bucket_name,
                    public=False  # Keep files private by default
                )
                if result.get("success"):
                    logger.info(f"Created bucket: {self.bucket_name}")
                    return True
                else:
                    logger.error(f"Failed to create bucket: {result.get('error')}")
                    return False
            except Exception as e:
                logger.error(f"Error creating bucket {self.bucket_name}: {str(e)}")
                return False

    def upload_file(
        self,
        file_obj: BinaryIO,
        object_key: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Upload a file to Supabase Storage"""
        try:
            # Ensure bucket exists
            if not self.ensure_bucket_exists():
                return {"success": False, "error": "Bucket creation failed"}

            # Read file content
            file_content = file_obj.read()

            # Upload to Supabase
            result = self.supabase_service.upload_file(
                bucket=self.bucket_name,
                file_path=object_key,
                file_content=file_content,
                content_type=content_type
            )

            if result.get("success"):
                return {
                    "success": True,
                    "object_key": object_key,
                    "url": result.get("public_url"),
                    "bucket": self.bucket_name,
                    "content_type": content_type,
                    "metadata": metadata or {}
                }
            else:
                return {"success": False, "error": result.get("error")}

        except Exception as e:
            logger.error(f"Error uploading file {object_key}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Upload bytes data to Supabase Storage"""
        from io import BytesIO
        file_obj = BytesIO(data)
        return self.upload_file(file_obj, object_key, content_type, metadata)

    def download_file(self, object_key: str) -> Optional[bytes]:
        """Download a file from Supabase Storage"""
        try:
            return self.supabase_service.download_file(self.bucket_name, object_key)
        except Exception as e:
            logger.error(f"Error downloading file {object_key}: {e}")
            return None

    def get_public_url(self, object_key: str) -> Optional[str]:
        """Get public URL for a file"""
        try:
            return self.supabase_service.client.storage.from_(self.bucket_name).get_public_url(object_key)
        except Exception as e:
            logger.error(f"Error getting public URL for {object_key}: {e}")
            return None

    def delete_file(self, object_key: str) -> bool:
        """Delete a file from Supabase Storage"""
        try:
            return self.supabase_service.delete_file(self.bucket_name, object_key)
        except Exception as e:
            logger.error(f"Error deleting file {object_key}: {e}")
            return False

    def list_objects(
        self,
        prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """List objects in storage with optional prefix"""
        try:
            files = self.supabase_service.list_files(self.bucket_name, prefix)

            objects = []
            for file_info in files:
                objects.append({
                    'key': file_info.get('name', ''),
                    'size': file_info.get('metadata', {}).get('size', 0),
                    'last_modified': file_info.get('created_at'),
                    'etag': file_info.get('id', '')
                })

            return objects
        except Exception as e:
            logger.error(f"Error listing objects with prefix {prefix}: {e}")
            return []

    def get_object_metadata(self, object_key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata"""
        try:
            # For Supabase, we need to implement metadata retrieval
            # This is a simplified version - in practice you might store metadata separately
            return {
                'content_type': 'application/octet-stream',  # Default
                'object_key': object_key,
                'bucket': self.bucket_name
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {object_key}: {e}")
            return None

    # StudySense-specific methods

    def store_kb_document(
        self,
        document_content: bytes,
        filename: str,
        document_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store a knowledge base document"""
        object_key = f"kb/{document_type}/{filename}"
        storage_metadata = {
            'document_type': document_type,
            'uploaded_at': datetime.utcnow().isoformat(),
            **{k: str(v) for k, v in metadata.items()}
        }

        content_type = 'application/pdf' if filename.endswith('.pdf') else 'text/plain'

        return self.upload_bytes(
            data=document_content,
            object_key=object_key,
            content_type=content_type,
            metadata=storage_metadata
        )

    def export_user_data(
        self,
        user_id: int,
        export_data: bytes,
        export_type: str = "full"
    ) -> Dict[str, Any]:
        """Export user data for GDPR/compliance"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"user_{user_id}_export_{export_type}_{timestamp}.json"
        object_key = f"exports/users/{user_id}/{filename}"

        metadata = {
            'user_id': str(user_id),
            'export_type': export_type,
            'exported_at': datetime.utcnow().isoformat(),
            'data_retention_days': str(settings.AUDIT_RETENTION_DAYS)
        }

        result = self.upload_bytes(
            data=export_data,
            object_key=object_key,
            content_type='application/json',
            metadata=metadata
        )

        # Schedule deletion after retention period (implementation depends on your needs)
        return result

    def store_anonymized_data(
        self,
        data: bytes,
        dataset_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store anonymized research data"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_type}_{timestamp}.parquet"
        object_key = f"research/anonymized/{dataset_type}/{filename}"

        storage_metadata = {
            'dataset_type': dataset_type,
            'created_at': datetime.utcnow().isoformat(),
            'anonymized': 'true',
            **{k: str(v) for k, v in metadata.items()}
        }

        return self.upload_bytes(
            data=data,
            object_key=object_key,
            content_type='application/octet-stream',
            metadata=storage_metadata
        )

    def store_report(
        self,
        user_id: int,
        report_data: bytes,
        report_type: str,
        period: str
    ) -> Dict[str, Any]:
        """Store user reports"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{period}_{timestamp}.pdf"
        object_key = f"reports/users/{user_id}/{report_type}/{filename}"

        metadata = {
            'user_id': str(user_id),
            'report_type': report_type,
            'period': period,
            'generated_at': datetime.utcnow().isoformat(),
            'retention_days': str(settings.DEFAULT_DATA_RETENTION_DAYS)
        }

        return self.upload_bytes(
            data=report_data,
            object_key=object_key,
            content_type='application/pdf',
            metadata=metadata
        )

    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired objects based on retention policies"""
        cleanup_counts = {}

        # This is a simplified implementation
        # In practice, you would:
        # 1. Query metadata for creation dates
        # 2. Compare with retention policies
        # 3. Delete expired files

        logger.info("Cleanup expired data: Simplified implementation")
        cleanup_counts['exports_deleted'] = 0  # Placeholder

        return cleanup_counts

    def health_check(self) -> Dict[str, Any]:
        """Check storage service health"""
        try:
            # Test bucket access
            bucket_exists = self.ensure_bucket_exists()

            if bucket_exists:
                # Test upload/download cycle
                test_data = b"health_check_test"
                test_key = "health_check/test.txt"

                upload_result = self.upload_bytes(test_data, test_key, "text/plain")
                if upload_result.get("success"):
                    downloaded_data = self.download_file(test_key)
                    self.delete_file(test_key)  # Clean up

                    return {
                        "status": "healthy",
                        "bucket_accessible": True,
                        "upload_download_test": downloaded_data == test_data
                    }

            return {
                "status": "unhealthy",
                "bucket_accessible": False,
                "error": "Storage bucket not accessible"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global storage instance
storage = StudySenseStorage()