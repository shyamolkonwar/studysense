from typing import Dict, List, Any, Optional, Set
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
import json
from enum import Enum
import weakref

logger = logging.getLogger(__name__)

class StreamType(str, Enum):
    """Types of streams supported"""
    WEBSOCKET = "websocket"
    SSE = "server_sent_events"

class StreamEvent:
    """Represents a streaming event"""

    def __init__(
        self,
        event_type: str,
        data: Any,
        timestamp: Optional[datetime] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.user_id = user_id
        self.session_id = session_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }

    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())

@dataclass
class StreamConnection:
    """Represents a streaming connection"""
    connection_id: str
    user_id: str
    session_id: str
    stream_type: StreamType
    connection: Any  # WebSocket or HTTP response object
    subscriptions: Set[str]
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

class StreamManager:
    """
    Manages all streaming connections and event broadcasting
    """

    def __init__(self):
        """Initialize stream manager"""
        self.connections: Dict[str, StreamConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.event_handlers: Dict[str, List[callable]] = {}
        self.connection_cleanup_interval = 300  # 5 minutes
        self._cleanup_task = None

        # Metrics
        self.total_connections = 0
        self.active_connections = 0
        self.events_sent = 0
        self.events_received = 0

        logger.info("Stream Manager initialized")

    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Stream cleanup task started")

    async def add_connection(
        self,
        connection_id: str,
        user_id: str,
        session_id: str,
        stream_type: StreamType,
        connection: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new streaming connection

        Args:
            connection_id: Unique connection identifier
            user_id: User identifier
            session_id: Session identifier
            stream_type: Type of stream (WebSocket or SSE)
            connection: Connection object
            metadata: Additional connection metadata

        Returns:
            True if connection added successfully
        """
        try:
            if connection_id in self.connections:
                logger.warning(f"Connection {connection_id} already exists")
                return False

            # Create stream connection object
            stream_conn = StreamConnection(
                connection_id=connection_id,
                user_id=user_id,
                session_id=session_id,
                stream_type=stream_type,
                connection=connection,
                subscriptions=set(),
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata=metadata or {}
            )

            # Store connection
            self.connections[connection_id] = stream_conn

            # Update user and session mappings
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)

            self.session_connections[session_id] = connection_id

            # Update metrics
            self.total_connections += 1
            self.active_connections += 1

            logger.info(f"Stream connection added: {connection_id} for user {user_id}")

            # Send welcome event
            await self.send_to_connection(
                connection_id,
                StreamEvent(
                    event_type="connection_established",
                    data={
                        "connection_id": connection_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "stream_type": stream_type.value
                    }
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add connection {connection_id}: {e}")
            return False

    async def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a streaming connection

        Args:
            connection_id: Connection identifier to remove

        Returns:
            True if connection removed successfully
        """
        try:
            if connection_id not in self.connections:
                logger.warning(f"Connection {connection_id} not found for removal")
                return False

            stream_conn = self.connections[connection_id]

            # Remove from user and session mappings
            if stream_conn.user_id in self.user_connections:
                self.user_connections[stream_conn.user_id].discard(connection_id)
                if not self.user_connections[stream_conn.user_id]:
                    del self.user_connections[stream_conn.user_id]

            if stream_conn.session_id in self.session_connections:
                del self.session_connections[stream_conn.session_id]

            # Close connection if it has a close method
            if hasattr(stream_conn.connection, 'close'):
                try:
                    await stream_conn.connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {connection_id}: {e}")

            # Remove from main dictionary
            del self.connections[connection_id]

            # Update metrics
            self.active_connections -= 1

            logger.info(f"Stream connection removed: {connection_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to remove connection {connection_id}: {e}")
            return False

    async def subscribe(
        self,
        connection_id: str,
        event_types: List[str]
    ) -> bool:
        """
        Subscribe a connection to specific event types

        Args:
            connection_id: Connection identifier
            event_types: List of event types to subscribe to

        Returns:
            True if subscription successful
        """
        try:
            if connection_id not in self.connections:
                logger.error(f"Cannot subscribe unknown connection: {connection_id}")
                return False

            stream_conn = self.connections[connection_id]
            stream_conn.subscriptions.update(event_types)

            logger.info(f"Connection {connection_id} subscribed to: {event_types}")

            # Send subscription confirmation
            await self.send_to_connection(
                connection_id,
                StreamEvent(
                    event_type="subscription_confirmed",
                    data={"subscribed_events": list(event_types)}
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to subscribe connection {connection_id}: {e}")
            return False

    async def unsubscribe(
        self,
        connection_id: str,
        event_types: Optional[List[str]] = None
    ) -> bool:
        """
        Unsubscribe a connection from event types

        Args:
            connection_id: Connection identifier
            event_types: Event types to unsubscribe from (None = all)

        Returns:
            True if unsubscription successful
        """
        try:
            if connection_id not in self.connections:
                return False

            stream_conn = self.connections[connection_id]

            if event_types is None:
                # Unsubscribe from all events
                stream_conn.subscriptions.clear()
                logger.info(f"Connection {connection_id} unsubscribed from all events")
            else:
                # Unsubscribe from specific events
                for event_type in event_types:
                    stream_conn.subscriptions.discard(event_type)
                logger.info(f"Connection {connection_id} unsubscribed from: {event_types}")

            # Send unsubscription confirmation
            await self.send_to_connection(
                connection_id,
                StreamEvent(
                    event_type="unsubscription_confirmed",
                    data={"unsubscribed_events": event_types or "all"}
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe connection {connection_id}: {e}")
            return False

    async def send_to_connection(
        self,
        connection_id: str,
        event: StreamEvent
    ) -> bool:
        """
        Send an event to a specific connection

        Args:
            connection_id: Target connection identifier
            event: Event to send

        Returns:
            True if event sent successfully
        """
        try:
            if connection_id not in self.connections:
                logger.warning(f"Cannot send to unknown connection: {connection_id}")
                return False

            stream_conn = self.connections[connection_id]

            # Update last activity
            stream_conn.last_activity = datetime.now()

            # Skip if connection is not subscribed to this event type
            if event.event_type not in stream_conn.subscriptions and "*" not in stream_conn.subscriptions:
                return False

            # Send event based on stream type
            if stream_conn.stream_type == StreamType.WEBSOCKET:
                return await self._send_websocket_event(stream_conn.connection, event)
            elif stream_conn.stream_type == StreamType.SSE:
                return await self._send_sse_event(stream_conn.connection, event)
            else:
                logger.error(f"Unknown stream type for {connection_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to send event to {connection_id}: {e}")
            return False

    async def broadcast_to_user(
        self,
        user_id: str,
        event: StreamEvent
    ) -> int:
        """
        Broadcast an event to all connections for a user

        Args:
            user_id: Target user identifier
            event: Event to broadcast

        Returns:
            Number of connections the event was sent to
        """
        try:
            connection_ids = self.user_connections.get(user_id, set())
            sent_count = 0

            for connection_id in list(connection_ids):
                if await self.send_to_connection(connection_id, event):
                    sent_count += 1

            logger.debug(f"Broadcast to user {user_id}: sent to {sent_count}/{len(connection_ids)} connections")
            return sent_count

        except Exception as e:
            logger.error(f"Failed to broadcast to user {user_id}: {e}")
            return 0

    async def broadcast_to_all(
        self,
        event: StreamEvent,
        exclude_connections: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast an event to all connections

        Args:
            event: Event to broadcast
            exclude_connections: Connections to exclude from broadcast

        Returns:
            Number of connections the event was sent to
        """
        try:
            sent_count = 0
            exclude_set = exclude_connections or set()

            for connection_id in self.connections:
                if connection_id not in exclude_set:
                    if await self.send_to_connection(connection_id, event):
                        sent_count += 1

            self.events_sent += sent_count
            logger.debug(f"Broadcast to all: sent to {sent_count} connections")

            return sent_count

        except Exception as e:
            logger.error(f"Failed to broadcast to all: {e}")
            return 0

    async def _send_websocket_event(self, websocket, event: StreamEvent) -> bool:
        """Send event via WebSocket"""
        try:
            message = {
                "type": "event",
                "payload": event.to_dict()
            }

            await websocket.send_json(message)
            return True

        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            return False

    async def _send_sse_event(self, response, event: StreamEvent) -> bool:
        """Send event via Server-Sent Events"""
        try:
            event_data = f"event: {event.event_type}\n"
            event_data += f"data: {json.dumps(event.data)}\n"
            event_data += f"timestamp: {event.timestamp.isoformat()}\n"
            event_data += "\n"

            await response.write(event_data)
            await response.drain()
            return True

        except Exception as e:
            logger.error(f"SSE send error: {e}")
            return False

    async def _periodic_cleanup(self):
        """Periodic cleanup of inactive connections"""
        while True:
            try:
                await asyncio.sleep(self.connection_cleanup_interval)
                await self._cleanup_inactive_connections()

            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def _cleanup_inactive_connections(self):
        """Remove inactive connections"""
        try:
            now = datetime.now()
            inactive_threshold = 300  # 5 minutes

            inactive_connections = []

            for connection_id, stream_conn in self.connections.items():
                time_inactive = (now - stream_conn.last_activity).total_seconds()
                if time_inactive > inactive_threshold:
                    inactive_connections.append(connection_id)

            # Remove inactive connections
            for connection_id in inactive_connections:
                logger.info(f"Removing inactive connection: {connection_id}")
                await self.remove_connection(connection_id)

            if inactive_connections:
                logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        connections_by_type = {
            "websocket": 0,
            "sse": 0
        }

        for stream_conn in self.connections.values():
            connections_by_type[stream_conn.stream_type.value] += 1

        return {
            "total_connections": len(self.connections),
            "active_connections": self.active_connections,
            "connections_by_type": connections_by_type,
            "users_connected": len(self.user_connections),
            "events_sent": self.events_sent,
            "events_received": self.events_received,
            "sessions_active": len(self.session_connections)
        }

    def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all connections for a user"""
        connection_ids = self.user_connections.get(user_id, set())
        connections = []

        for connection_id in connection_ids:
            if connection_id in self.connections:
                stream_conn = self.connections[connection_id]
                connections.append({
                    "connection_id": connection_id,
                    "stream_type": stream_conn.stream_type.value,
                    "session_id": stream_conn.session_id,
                    "subscriptions": list(stream_conn.subscriptions),
                    "created_at": stream_conn.created_at.isoformat(),
                    "last_activity": stream_conn.last_activity.isoformat(),
                    "metadata": stream_conn.metadata
                })

        return connections

    async def shutdown(self):
        """Shutdown the stream manager"""
        logger.info("Shutting down Stream Manager")

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.remove_connection(connection_id)

        logger.info("Stream Manager shutdown complete")

# Global stream manager instance
stream_manager = StreamManager()