"""
Phase 5: Notifications API Endpoints
REST API endpoints for notification management
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import logging

from ...notifications.notification_service import notification_service, NotificationStatus, NotificationPriority
from ...notifications.notification_channels import InAppChannel
from ...core.auth import get_current_user
from ...schemas.notification import (
    NotificationResponse,
    NotificationListResponse,
    NotificationPreferencesRequest,
    NotificationPreferencesResponse,
    DeliveryStatsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/notifications", tags=["notifications"])

@router.get("/", response_model=NotificationListResponse)
async def get_notifications(
    current_user: Dict[str, Any] = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by notification status"),
    channel: Optional[str] = Query(None, description="Filter by notification channel"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of notifications to return"),
    offset: int = Query(0, ge=0, description="Number of notifications to skip"),
    unread_only: bool = Query(False, description="Only return unread notifications")
):
    """Get notifications for the current user"""
    try:
        user_id = current_user["id"]

        # Get in-app notifications
        in_app_channel = None
        for channel_type, channel_instance in notification_service.channels.items():
            if isinstance(channel_instance, InAppChannel):
                in_app_channel = channel_instance
                break

        if not in_app_channel:
            return NotificationListResponse(
                notifications=[],
                total=0,
                limit=limit,
                offset=offset
            )

        # Get notifications
        notifications = in_app_channel.get_user_notifications(user_id, unread_only)

        # Apply filters
        if status:
            notifications = [n for n in notifications if n.get("status") == status]

        if channel:
            notifications = [n for n in notifications if n.get("channel") == channel]

        # Apply pagination
        total = len(notifications)
        notifications = notifications[offset:offset + limit]

        # Convert to response format
        notification_responses = [
            NotificationResponse(
                id=notification["id"],
                channel=notification.get("channel", "in_app"),
                priority=notification.get("priority", "medium"),
                subject=notification.get("subject", ""),
                content=notification.get("content", ""),
                created_at=notification["created_at"],
                read=notification["read"],
                metadata=notification.get("metadata", {})
            )
            for notification in notifications
        ]

        return NotificationListResponse(
            notifications=notification_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error getting notifications for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")

@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Mark a notification as read"""
    try:
        user_id = current_user["id"]

        # Get in-app channel
        in_app_channel = None
        for channel_type, channel_instance in notification_service.channels.items():
            if isinstance(channel_instance, InAppChannel):
                in_app_channel = channel_instance
                break

        if not in_app_channel:
            raise HTTPException(status_code=404, detail="In-app notification channel not found")

        # Mark as read
        success = in_app_channel.mark_notification_read(user_id, notification_id)

        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")

        return {"message": "Notification marked as read", "notification_id": notification_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@router.post("/mark-all-read")
async def mark_all_notifications_read(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Mark all notifications as read for the current user"""
    try:
        user_id = current_user["id"]

        # Get in-app channel
        in_app_channel = None
        for channel_type, channel_instance in notification_service.channels.items():
            if isinstance(channel_instance, InAppChannel):
                in_app_channel = channel_instance
                break

        if not in_app_channel:
            raise HTTPException(status_code=404, detail="In-app notification channel not found")

        # Mark all as read
        count = in_app_channel.mark_all_read(user_id)

        return {"message": f"Marked {count} notifications as read", "count": count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notifications as read")

@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get notification preferences for the current user"""
    try:
        user_id = current_user["id"]
        preferences = notification_service.get_user_preferences(user_id)

        return NotificationPreferencesResponse(
            channels=preferences["channels"],
            frequency=preferences["frequency"]
        )

    except Exception as e:
        logger.error(f"Error getting notification preferences for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notification preferences")

@router.put("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferencesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update notification preferences for the current user"""
    try:
        user_id = current_user["id"]

        # Merge with existing preferences
        current_prefs = notification_service.get_user_preferences(user_id)

        updated_prefs = {
            "channels": {
                **current_prefs.get("channels", {}),
                **preferences.channels.dict() if preferences.channels else {}
            },
            "frequency": {
                **current_prefs.get("frequency", {}),
                **preferences.frequency.dict() if preferences.frequency else {}
            }
        }

        # Update preferences
        notification_service.set_user_preferences(user_id, updated_prefs)

        return {"message": "Notification preferences updated successfully"}

    except Exception as e:
        logger.error(f"Error updating notification preferences for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update notification preferences")

@router.get("/stats", response_model=DeliveryStatsResponse)
async def get_notification_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back")
):
    """Get notification delivery statistics for the current user"""
    try:
        time_range = timedelta(days=days)
        stats = notification_service.get_delivery_stats(
            user_id=current_user["id"],
            time_range=time_range
        )

        return DeliveryStatsResponse(
            total=stats["total"],
            delivered=stats["delivered"],
            failed=stats["failed"],
            delivery_rate=stats["delivery_rate"],
            channel_breakdown=stats["channel_breakdown"],
            time_range_hours=stats["time_range_hours"]
        )

    except Exception as e:
        logger.error(f"Error getting notification stats for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notification statistics")

@router.post("/test")
async def test_notification(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    channel: str = Query("push", description="Notification channel to test"),
    message: str = Query("Test notification", description="Test message content")
):
    """Send a test notification"""
    try:
        user_id = current_user["id"]

        # Get user's contact information from their profile
        user_email = current_user.get("email", f"user_{user_id}@example.com")
        user_phone = current_user.get("phone", None)

        # Send test notification
        notification_id = await notification_service.send_notification(
            user_id=user_id,
            channel=channel,
            subject="Test Notification",
            content=message,
            priority="low",
            metadata={
                "test": True,
                "email": user_email,
                "phone": user_phone
            }
        )

        return {
            "message": "Test notification sent successfully",
            "notification_id": notification_id,
            "channel": channel
        }

    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test notification")

# Admin endpoints
@router.get("/admin/stats")
async def get_admin_notification_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    days: int = Query(7, ge=1, le=90, description="Number of days to look back")
):
    """Get admin notification statistics"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        time_range = timedelta(days=days)
        stats = notification_service.get_delivery_stats(time_range=time_range)

        return {
            "overall_stats": stats,
            "active_channels": list(notification_service.channels.keys()),
            "pending_messages": len(notification_service.pending_messages),
            "total_users": len(notification_service.user_preferences)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin notification stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notification statistics")

@router.post("/admin/broadcast")
async def broadcast_notification(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    channel: str = Query("push", description="Notification channel"),
    subject: str = Query(..., description="Broadcast subject"),
    content: str = Query(..., description="Broadcast content"),
    priority: str = Query("medium", description="Notification priority")
):
    """Broadcast notification to all users (admin only)"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        # Get all users (simplified - in production would query database)
        user_ids = list(notification_service.user_preferences.keys())

        if not user_ids:
            return {"message": "No users found to broadcast to"}

        # Prepare broadcast notifications
        notifications = []
        for user_id in user_ids[:100]:  # Limit to 100 users for safety
            notifications.append({
                "user_id": user_id,
                "channel": channel,
                "subject": subject,
                "content": content,
                "priority": priority,
                "metadata": {
                    "broadcast": True,
                    "sent_by": current_user["id"],
                    "sent_at": datetime.now().isoformat()
                }
            })

        # Send in background
        background_tasks.add_task(
            notification_service.send_bulk_notifications,
            notifications
        )

        return {
            "message": f"Broadcast scheduled for {len(notifications)} users",
            "channel": channel,
            "subject": subject
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error broadcasting notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast notification")