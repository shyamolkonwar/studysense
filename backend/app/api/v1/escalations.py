"""
Phase 5: Escalations API Endpoints
REST API endpoints for escalation management
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import logging

from ...alerts.escalation_manager import escalation_manager, EscalationStatus, EscalationLevel
from ...core.auth import get_current_user
from ...schemas.escalation import (
    EscalationResponse,
    EscalationListResponse,
    EscalationCancelRequest
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/escalations", tags=["escalations"])

@router.get("/", response_model=EscalationListResponse)
async def get_escalations(
    current_user: Dict[str, Any] = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by escalation status"),
    level: Optional[str] = Query(None, description="Filter by escalation level"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of escalations to return"),
    offset: int = Query(0, ge=0, description="Number of escalations to skip")
):
    """Get escalations for the current user"""
    try:
        user_id = current_user["id"]

        # Get escalations based on filters
        escalations = escalation_manager.get_active_escalations(user_id)

        # Apply filters
        if status:
            escalations = [e for e in escalations if e.status.value == status]

        if level:
            escalations = [e for e in escalations if e.current_level.value == level]

        # Apply pagination
        total = len(escalations)
        escalations = escalations[offset:offset + limit]

        # Convert to response format
        escalation_responses = [
            EscalationResponse(
                id=escalation.id,
                alert_id=escalation.alert_id,
                protocol_id=escalation.protocol_id,
                current_level=escalation.current_level.value,
                status=escalation.status.value,
                started_at=escalation.started_at,
                completed_at=escalation.completed_at,
                steps_completed=escalation.steps_completed,
                next_step_time=escalation.next_step_time,
                metadata=escalation.metadata
            )
            for escalation in escalations
        ]

        return EscalationListResponse(
            escalations=escalation_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error getting escalations for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalations")

@router.get("/history", response_model=EscalationListResponse)
async def get_escalation_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of escalations to return"),
    offset: int = Query(0, ge=0, description="Number of escalations to skip")
):
    """Get escalation history for the current user"""
    try:
        user_id = current_user["id"]

        # Get escalation history
        escalations = escalation_manager.get_escalation_history(user_id, limit=limit + offset)

        # Apply pagination
        total = len(escalations)
        escalations = escalations[offset:offset + limit]

        # Convert to response format
        escalation_responses = [
            EscalationResponse(
                id=escalation.id,
                alert_id=escalation.alert_id,
                protocol_id=escalation.protocol_id,
                current_level=escalation.current_level.value,
                status=escalation.status.value,
                started_at=escalation.started_at,
                completed_at=escalation.completed_at,
                steps_completed=escalation.steps_completed,
                next_step_time=escalation.next_step_time,
                metadata=escalation.metadata
            )
            for escalation in escalations
        ]

        return EscalationListResponse(
            escalations=escalation_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error getting escalation history for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalation history")

@router.post("/{escalation_id}/cancel")
async def cancel_escalation(
    escalation_id: str,
    request: EscalationCancelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel an active escalation"""
    try:
        user_id = current_user["id"]

        # Verify user owns the escalation
        escalation = escalation_manager.active_escalations.get(escalation_id)
        if not escalation or escalation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Escalation not found")

        # Cancel escalation
        success = escalation_manager.cancel_escalation(escalation_id, cancelled_by=user_id)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel escalation")

        return {
            "message": "Escalation cancelled successfully",
            "escalation_id": escalation_id,
            "reason": request.reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling escalation {escalation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel escalation")

@router.get("/stats")
async def get_escalation_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get escalation statistics for the current user"""
    try:
        user_id = current_user["id"]

        # Get user's escalations
        active_escalations = escalation_manager.get_active_escalations(user_id)
        escalation_history = escalation_manager.get_escalation_history(user_id, limit=1000)

        # Calculate statistics
        stats = {
            "active_escalations": len(active_escalations),
            "total_escalations": len(escalation_history),
            "escalations_by_level": {},
            "escalations_by_status": {},
            "recent_trend": [],
            "completion_rate": 0.0
        }

        # Level breakdown
        for escalation in active_escalations:
            level = escalation.current_level.value
            stats["escalations_by_level"][level] = stats["escalations_by_level"].get(level, 0) + 1

        # Status breakdown
        for escalation in escalation_history:
            status = escalation.status.value
            stats["escalations_by_status"][status] = stats["escalations_by_status"].get(status, 0) + 1

        # Calculate completion rate
        completed = len([e for e in escalation_history if e.status == EscalationStatus.COMPLETED])
        if escalation_history:
            stats["completion_rate"] = completed / len(escalation_history)

        # Recent trend (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_escalations = [e for e in escalation_history if e.started_at > thirty_days_ago]

        daily_counts = {}
        for escalation in recent_escalations:
            day = escalation.started_at.date().isoformat()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        # Fill missing days
        for i in range(30):
            day = (datetime.now() - timedelta(days=i)).date().isoformat()
            if day not in daily_counts:
                daily_counts[day] = 0

        stats["recent_trend"] = [
            {"date": day, "count": daily_counts.get(day, 0)}
            for day in sorted(daily_counts.keys())
        ]

        return stats

    except Exception as e:
        logger.error(f"Error getting escalation stats for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalation statistics")

# Crisis management endpoints
@router.post("/crisis")
async def trigger_crisis_escalation(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    crisis_data: Dict[str, Any] = None
):
    """Trigger immediate crisis escalation"""
    try:
        user_id = current_user["id"]

        # Create crisis alert
        crisis_alert = {
            "id": f"crisis_{user_id}_{int(datetime.now().timestamp())}",
            "rule_id": "crisis_manual_trigger",
            "user_id": user_id,
            "severity": "critical",
            "status": "active",
            "title": "🚨 CRISIS: Manual Crisis Alert Triggered",
            "message": crisis_data.get("message", "User has triggered a crisis alert and needs immediate support."),
            "triggered_at": datetime.now(),
            "risk_score": 0.95,
            "risk_level": "crisis",
            "context": {
                "manual_trigger": True,
                "crisis_data": crisis_data or {},
                "timestamp": datetime.now().isoformat()
            }
        }

        # Process crisis escalation in background
        background_tasks.add_task(
            escalation_manager.process_alert_for_escalation,
            crisis_alert,
            {"priority": "critical", "bypass_consent": True}
        )

        return {
            "message": "Crisis escalation triggered successfully",
            "alert_id": crisis_alert["id"],
            "immediate_actions": [
                "Call 988 (Suicide Prevention Lifeline)",
                "Text HOME to 741741",
                "Call 911 for immediate emergency"
            ]
        }

    except Exception as e:
        logger.error(f"Error triggering crisis escalation for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger crisis escalation")

@router.get("/crisis/resources")
async def get_crisis_resources():
    """Get crisis resources and helplines"""
    try:
        resources = escalation_manager.crisis_resources

        return {
            "hotlines": resources["hotlines"],
            "resources": resources["resources"],
            "immediate_steps": [
                "1. Call or text a crisis hotline now",
                "2. Remove yourself from immediate danger if possible",
                "3. Contact someone you trust",
                "4. If in immediate danger, call 911",
                "5. Stay on the phone with crisis services until help arrives"
            ],
            "disclaimer": "If you are in immediate danger, please call emergency services (911) right now."
        }

    except Exception as e:
        logger.error(f"Error getting crisis resources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve crisis resources")

# Admin endpoints
@router.get("/admin/all", response_model=EscalationListResponse)
async def get_all_escalations(
    current_user: Dict[str, Any] = Depends(get_current_user),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by escalation status"),
    level: Optional[str] = Query(None, description="Filter by escalation level"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of escalations to return"),
    offset: int = Query(0, ge=0, description="Number of escalations to skip")
):
    """Get all escalations (admin only)"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        # Get all escalations
        escalations = escalation_manager.get_active_escalations()

        # Apply filters
        if user_id:
            escalations = [e for e in escalations if e.user_id == user_id]

        if status:
            escalations = [e for e in escalations if e.status.value == status]

        if level:
            escalations = [e for e in escalations if e.current_level.value == level]

        # Apply pagination
        total = len(escalations)
        escalations = escalations[offset:offset + limit]

        # Convert to response format
        escalation_responses = [
            EscalationResponse(
                id=escalation.id,
                alert_id=escalation.alert_id,
                user_id=escalation.user_id,  # Include user_id for admin view
                protocol_id=escalation.protocol_id,
                current_level=escalation.current_level.value,
                status=escalation.status.value,
                started_at=escalation.started_at,
                completed_at=escalation.completed_at,
                steps_completed=escalation.steps_completed,
                next_step_time=escalation.next_step_time,
                metadata=escalation.metadata
            )
            for escalation in escalations
        ]

        return EscalationListResponse(
            escalations=escalation_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all escalations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalations")

@router.get("/admin/stats")
async def get_admin_escalation_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get admin escalation statistics"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        return escalation_manager.get_escalation_stats()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin escalation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalation statistics")

@router.post("/admin/{escalation_id}/cancel")
async def admin_cancel_escalation(
    escalation_id: str,
    request: EscalationCancelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel an escalation (admin only)"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        # Cancel escalation
        success = escalation_manager.cancel_escalation(
            escalation_id,
            cancelled_by=f"admin_{current_user['id']}"
        )

        if not success:
            raise HTTPException(status_code=404, detail="Escalation not found")

        return {
            "message": "Escalation cancelled successfully by admin",
            "escalation_id": escalation_id,
            "cancelled_by": current_user["id"],
            "reason": request.reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling escalation {escalation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel escalation")