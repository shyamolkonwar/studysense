"""
Phase 5: Alerts API Endpoints
REST API endpoints for alert management and escalation
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import logging

from ...alerts.alert_engine import alert_engine, Alert, AlertStatus, AlertSeverity
from ...alerts.escalation_manager import escalation_manager, EscalationStatus
from ...core.auth import get_current_user
from ...schemas.alert import (
    AlertResponse,
    AlertListResponse,
    AlertAcknowledgeRequest,
    AlertPolicyResponse,
    EscalationResponse,
    EscalationListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])

@router.get("/", response_model=AlertListResponse)
async def get_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip")
):
    """Get alerts for the current user"""
    try:
        user_id = current_user["id"]

        # Get alerts based on filters
        alerts = alert_engine.get_active_alerts(user_id)

        # Apply filters
        if status:
            alerts = [alert for alert in alerts if alert.status.value == status]

        if severity:
            alerts = [alert for alert in alerts if alert.severity.value == severity]

        # Apply pagination
        total = len(alerts)
        alerts = alerts[offset:offset + limit]

        # Convert to response format
        alert_responses = [
            AlertResponse(
                id=alert.id,
                rule_id=alert.rule_id,
                severity=alert.severity.value,
                status=alert.status.value,
                title=alert.title,
                message=alert.message,
                triggered_at=alert.triggered_at,
                acknowledged_at=alert.acknowledged_at,
                resolved_at=alert.resolved_at,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                context=alert.context
            )
            for alert in alerts
        ]

        return AlertListResponse(
            alerts=alert_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error getting alerts for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.get("/history", response_model=AlertListResponse)
async def get_alert_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip")
):
    """Get alert history for the current user"""
    try:
        user_id = current_user["id"]

        # Get alert history
        alerts = alert_engine.get_alert_history(user_id, limit=limit + offset)

        # Apply pagination
        total = len(alerts)
        alerts = alerts[offset:offset + limit]

        # Convert to response format
        alert_responses = [
            AlertResponse(
                id=alert.id,
                rule_id=alert.rule_id,
                severity=alert.severity.value,
                status=alert.status.value,
                title=alert.title,
                message=alert.message,
                triggered_at=alert.triggered_at,
                acknowledged_at=alert.acknowledged_at,
                resolved_at=alert.resolved_at,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                context=alert.context
            )
            for alert in alerts
        ]

        return AlertListResponse(
            alerts=alert_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Error getting alert history for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert history")

@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        user_id = current_user["id"]

        # Verify user owns the alert
        alert = alert_engine.active_alerts.get(alert_id)
        if not alert or alert.user_id != user_id:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Acknowledge alert
        success = alert_engine.acknowledge_alert(alert_id, acknowledged_by=user_id)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to acknowledge alert")

        return {"message": "Alert acknowledged successfully", "alert_id": alert_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Resolve an alert"""
    try:
        user_id = current_user["id"]

        # Verify user owns the alert
        alert = alert_engine.active_alerts.get(alert_id)
        if not alert or alert.user_id != user_id:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Resolve alert
        success = alert_engine.resolve_alert(alert_id, resolved_by=user_id)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to resolve alert")

        return {"message": "Alert resolved successfully", "alert_id": alert_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")

@router.get("/policies", response_model=List[AlertPolicyResponse])
async def get_alert_policies(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get alert policies"""
    try:
        policies = []

        for policy_id, policy in alert_engine.policies.items():
            policy_response = AlertPolicyResponse(
                id=policy.id,
                name=policy.name,
                description=policy.description,
                rules_count=len(policy.rules),
                notification_channels=policy.notification_channels,
                escalation_enabled=policy.escalation_enabled,
                user_consent_required=policy.user_consent_required
            )
            policies.append(policy_response)

        return policies

    except Exception as e:
        logger.error(f"Error getting alert policies: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert policies")

@router.get("/stats")
async def get_alert_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get alert statistics for the current user"""
    try:
        user_id = current_user["id"]

        # Get user's alerts
        active_alerts = alert_engine.get_active_alerts(user_id)
        alert_history = alert_engine.get_alert_history(user_id, limit=1000)

        # Calculate statistics
        stats = {
            "active_alerts": len(active_alerts),
            "total_alerts": len(alert_history),
            "alerts_by_severity": {},
            "alerts_by_status": {},
            "recent_trend": []
        }

        # Severity breakdown
        for alert in active_alerts:
            severity = alert.severity.value
            stats["alerts_by_severity"][severity] = stats["alerts_by_severity"].get(severity, 0) + 1

        # Status breakdown
        for alert in alert_history:
            status = alert.status.value
            stats["alerts_by_status"][status] = stats["alerts_by_status"].get(status, 0) + 1

        # Recent trend (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_alerts = [alert for alert in alert_history if alert.triggered_at > seven_days_ago]

        daily_counts = {}
        for alert in recent_alerts:
            day = alert.triggered_at.date().isoformat()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        # Fill missing days
        for i in range(7):
            day = (datetime.now() - timedelta(days=i)).date().isoformat()
            if day not in daily_counts:
                daily_counts[day] = 0

        stats["recent_trend"] = [
            {"date": day, "count": daily_counts.get(day, 0)}
            for day in sorted(daily_counts.keys())
        ]

        return stats

    except Exception as e:
        logger.error(f"Error getting alert stats for user {current_user['id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert statistics")

# Admin endpoints
@router.get("/admin/all", response_model=AlertListResponse)
async def get_all_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip")
):
    """Get all alerts (admin only)"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        # Get all alerts
        alerts = alert_engine.get_active_alerts()

        # Apply filters
        if user_id:
            alerts = [alert for alert in alerts if alert.user_id == user_id]

        if status:
            alerts = [alert for alert in alerts if alert.status.value == status]

        if severity:
            alerts = [alert for alert in alerts if alert.severity.value == severity]

        # Apply pagination
        total = len(alerts)
        alerts = alerts[offset:offset + limit]

        # Convert to response format
        alert_responses = [
            AlertResponse(
                id=alert.id,
                rule_id=alert.rule_id,
                user_id=alert.user_id,  # Include user_id for admin view
                severity=alert.severity.value,
                status=alert.status.value,
                title=alert.title,
                message=alert.message,
                triggered_at=alert.triggered_at,
                acknowledged_at=alert.acknowledged_at,
                resolved_at=alert.resolved_at,
                risk_score=alert.risk_score,
                risk_level=alert.risk_level,
                context=alert.context
            )
            for alert in alerts
        ]

        return AlertListResponse(
            alerts=alert_responses,
            total=total,
            limit=limit,
            offset=offset
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.get("/admin/stats")
async def get_admin_alert_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get admin alert statistics"""
    try:
        # Verify admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permissions required")

        return alert_engine.get_policy_stats()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin alert stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alert statistics")