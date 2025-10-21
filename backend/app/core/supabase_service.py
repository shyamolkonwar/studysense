"""
Supabase service layer for StudySense
Provides additional Supabase-specific functionality beyond direct database access
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.core.config import settings, get_supabase_client

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for Supabase-specific operations"""

    def __init__(self):
        self.client = get_supabase_client()

    def is_available(self) -> bool:
        """Check if Supabase client is available"""
        return self.client is not None

    def create_user_auth(self, email: str, password: str, user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new user in Supabase Auth"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": user_metadata or {}
                }
            })

            if response.user:
                return {
                    "success": True,
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "created_at": response.user.created_at
                }
            else:
                return {
                    "success": False,
                    "error": "No user returned from Supabase"
                }

        except Exception as e:
            logger.error(f"Error creating Supabase auth user: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user with Supabase Auth"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user:
                return {
                    "success": True,
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at
                }
            else:
                return {
                    "success": False,
                    "error": "Authentication failed"
                }

        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def sign_out_user(self, access_token: str) -> bool:
        """Sign out user from Supabase Auth"""
        if not self.client:
            return False

        try:
            # Set the auth header
            self.client.auth.set_session(access_token)
            response = self.client.auth.sign_out()
            return True
        except Exception as e:
            logger.error(f"Error signing out user: {str(e)}")
            return False

    def reset_password(self, email: str) -> Dict[str, Any]:
        """Send password reset email"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.auth.reset_password_for_email(email)
            return {
                "success": True,
                "message": "Password reset email sent"
            }
        except Exception as e:
            logger.error(f"Error sending password reset: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def upload_file(self, bucket: str, file_path: str, file_content: bytes, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Upload file to Supabase Storage"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.storage.from_(bucket).upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": content_type}
            )

            if response.data:
                public_url = self.client.storage.from_(bucket).get_public_url(file_path)
                return {
                    "success": True,
                    "path": file_path,
                    "public_url": public_url,
                    "bucket": bucket,
                    "content_type": content_type
                }
            else:
                return {
                    "success": False,
                    "error": "Upload failed"
                }

        except Exception as e:
            logger.error(f"Error uploading file to Supabase: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def download_file(self, bucket: str, file_path: str) -> Optional[bytes]:
        """Download file from Supabase Storage"""
        if not self.client:
            return None

        try:
            response = self.client.storage.from_(bucket).download(file_path)
            return response
        except Exception as e:
            logger.error(f"Error downloading file from Supabase: {str(e)}")
            return None

    def delete_file(self, bucket: str, file_path: str) -> bool:
        """Delete file from Supabase Storage"""
        if not self.client:
            return False

        try:
            response = self.client.storage.from_(bucket).remove([file_path])
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error deleting file from Supabase: {str(e)}")
            return False

    def create_bucket(self, bucket_name: str, public: bool = False) -> Dict[str, Any]:
        """Create a new storage bucket"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.storage.create_bucket(
                id=bucket_name,
                options={"public": public}
            )

            return {
                "success": True,
                "bucket_name": bucket_name
            }
        except Exception as e:
            logger.error(f"Error creating bucket: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_files(self, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in storage bucket"""
        if not self.client:
            return []

        try:
            response = self.client.storage.from_(bucket).list(prefix)
            return response
        except Exception as e:
            logger.error(f"Error listing files in Supabase: {str(e)}")
            return []

    def execute_rpc(self, function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a database function (RPC)"""
        if not self.client:
            raise Exception("Supabase client not available")

        try:
            response = self.client.rpc(function_name, params)
            return {
                "success": True,
                "data": response.data
            }
        except Exception as e:
            logger.error(f"Error executing RPC function: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Supabase services"""
        if not self.client:
            return {
                "available": False,
                "error": "Supabase client not configured"
            }

        try:
            # Test database connection
            start_time = datetime.utcnow()
            result = self.client.table('users').select('id').limit(1).execute()
            db_response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Test storage (if available)
            storage_status = "not_configured"
            if settings.SUPABASE_SERVICE_ROLE_KEY:
                try:
                    buckets = self.client.storage.list_buckets()
                    storage_status = "available"
                except:
                    storage_status = "error"

            return {
                "available": True,
                "database": {
                    "status": "connected",
                    "response_time_ms": db_response_time
                },
                "storage": {
                    "status": storage_status
                },
                "auth": {
                    "status": "configured"
                }
            }
        except Exception as e:
            logger.error(f"Supabase health check failed: {str(e)}")
            return {
                "available": False,
                "error": str(e)
            }


# Global service instance
supabase_service = SupabaseService()