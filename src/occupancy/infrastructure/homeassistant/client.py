"""Home Assistant WebSocket and REST client."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession, WSMsgType

from ...config.settings import Settings
from ...domain.exceptions import (
    SensorDataException,
    SensorTimeoutException,
    InvalidSensorReadingException,
)

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for Home Assistant WebSocket and REST API integration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: Optional[ClientSession] = None
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self.message_id = 1
        self.entity_callbacks: Dict[str, List[Callable]] = {}
        self.subscribed_entities: Set[str] = set()
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    async def __aenter__(self) -> "HomeAssistantClient":
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self) -> None:
        """Establish connections to Home Assistant."""
        try:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.settings.ha_token}"},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test REST API connection
            await self._test_api_connection()
            
            # Establish WebSocket connection
            await self._connect_websocket()
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully connected to Home Assistant")
            
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            await self.disconnect()
            raise SensorDataException(f"Home Assistant connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close all connections to Home Assistant."""
        self.is_connected = False
        
        if self.ws_connection and not self.ws_connection.closed:
            await self.ws_connection.close()
            
        if self.session:
            await self.session.close()
            
        self.ws_connection = None
        self.session = None
        logger.info("Disconnected from Home Assistant")
        
    async def _test_api_connection(self) -> None:
        """Test REST API connection."""
        if not self.session:
            raise SensorDataException("No session available")
            
        url = urljoin(self.settings.ha_url, "/api/")
        
        async with self.session.get(url) as response:
            if response.status != 200:
                raise SensorDataException(
                    f"HA API test failed: HTTP {response.status}"
                )
            
            data = await response.json()
            if data.get("message") != "API running.":
                raise SensorDataException("HA API not running")
                
    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection."""
        if not self.session:
            raise SensorDataException("No session available")
            
        ws_url = self.settings.ha_url.replace("http", "ws") + "/api/websocket"
        
        self.ws_connection = await self.session.ws_connect(ws_url)
        
        # Handle authentication flow
        auth_msg = await self.ws_connection.receive()
        if auth_msg.type == WSMsgType.TEXT:
            auth_data = json.loads(auth_msg.data)
            if auth_data.get("type") != "auth_required":
                raise SensorDataException("Unexpected WebSocket auth message")
                
        # Send authentication
        await self.ws_connection.send_str(json.dumps({
            "type": "auth",
            "access_token": self.settings.ha_token
        }))
        
        # Wait for auth response
        auth_response = await self.ws_connection.receive()
        if auth_response.type == WSMsgType.TEXT:
            auth_result = json.loads(auth_response.data)
            if auth_result.get("type") != "auth_ok":
                raise SensorDataException(f"WebSocket auth failed: {auth_result}")
                
        # Start message processing task
        asyncio.create_task(self._process_websocket_messages())
        
    async def _process_websocket_messages(self) -> None:
        """Process incoming WebSocket messages."""
        if not self.ws_connection:
            return
            
        try:
            async for msg in self.ws_connection:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from WebSocket: {msg.data}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    logger.info("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket message processing error: {e}")
        finally:
            self.is_connected = False
            await self._attempt_reconnect()
            
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")
        
        if msg_type == "event":
            event_data = data.get("event", {})
            if event_data.get("event_type") == "state_changed":
                await self._handle_state_change(event_data.get("data", {}))
                
    async def _handle_state_change(self, state_data: Dict[str, Any]) -> None:
        """Handle state change event."""
        entity_id = state_data.get("entity_id")
        
        if entity_id in self.entity_callbacks:
            for callback in self.entity_callbacks[entity_id]:
                try:
                    await callback(state_data)
                except Exception as e:
                    logger.error(f"Error in state change callback for {entity_id}: {e}")
                    
    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to Home Assistant."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
            
        self.reconnect_attempts += 1
        backoff_seconds = min(30, 2 ** self.reconnect_attempts)
        
        logger.info(f"Attempting reconnection #{self.reconnect_attempts} in {backoff_seconds}s")
        await asyncio.sleep(backoff_seconds)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            await self._attempt_reconnect()
            
    async def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get current state of an entity via REST API."""
        if not self.session:
            raise SensorDataException("No session available")
            
        url = urljoin(self.settings.ha_url, f"/api/states/{entity_id}")
        
        try:
            async with self.session.get(url) as response:
                if response.status == 404:
                    raise InvalidSensorReadingException(
                        f"Entity not found: {entity_id}",
                        sensor_entity=entity_id
                    )
                    
                if response.status != 200:
                    raise SensorDataException(
                        f"Failed to get entity state: HTTP {response.status}"
                    )
                    
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise SensorDataException(f"HTTP error getting entity state: {e}")
            
    async def get_entity_history(
        self, 
        entity_id: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get historical states for an entity."""
        if not self.session:
            raise SensorDataException("No session available")
            
        params = {"filter_entity_id": entity_id}
        
        if end_time:
            url = urljoin(
                self.settings.ha_url,
                f"/api/history/period/{start_time.isoformat()}"
            )
            params["end_time"] = end_time.isoformat()
        else:
            url = urljoin(
                self.settings.ha_url,
                f"/api/history/period/{start_time.isoformat()}"
            )
            
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise SensorDataException(
                        f"Failed to get entity history: HTTP {response.status}"
                    )
                    
                history_data = await response.json()
                
                # HA returns list of lists, we want the first list (for our entity)
                if history_data and len(history_data) > 0:
                    return history_data[0]
                return []
                
        except aiohttp.ClientError as e:
            raise SensorDataException(f"HTTP error getting entity history: {e}")
            
    async def subscribe_to_entity(
        self, 
        entity_id: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to state changes for an entity."""
        if entity_id not in self.entity_callbacks:
            self.entity_callbacks[entity_id] = []
            
        self.entity_callbacks[entity_id].append(callback)
        
        # Subscribe to entity if not already subscribed
        if entity_id not in self.subscribed_entities:
            await self._subscribe_websocket_entity(entity_id)
            self.subscribed_entities.add(entity_id)
            
    async def _subscribe_websocket_entity(self, entity_id: str) -> None:
        """Subscribe to entity state changes via WebSocket."""
        if not self.ws_connection:
            raise SensorDataException("No WebSocket connection")
            
        subscribe_msg = {
            "id": self.message_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
            "entity_id": entity_id
        }
        
        await self.ws_connection.send_str(json.dumps(subscribe_msg))
        self.message_id += 1
        
    async def unsubscribe_from_entity(
        self, 
        entity_id: str, 
        callback: Optional[Callable] = None
    ) -> None:
        """Unsubscribe from entity state changes."""
        if entity_id not in self.entity_callbacks:
            return
            
        if callback:
            # Remove specific callback
            if callback in self.entity_callbacks[entity_id]:
                self.entity_callbacks[entity_id].remove(callback)
        else:
            # Remove all callbacks for entity
            self.entity_callbacks[entity_id].clear()
            
        # If no more callbacks, unsubscribe from WebSocket
        if not self.entity_callbacks[entity_id]:
            del self.entity_callbacks[entity_id]
            if entity_id in self.subscribed_entities:
                self.subscribed_entities.remove(entity_id)
                
    async def get_entities_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Get all entities matching a pattern."""
        if not self.session:
            raise SensorDataException("No session available")
            
        url = urljoin(self.settings.ha_url, "/api/states")
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise SensorDataException(
                        f"Failed to get entities: HTTP {response.status}"
                    )
                    
                all_entities = await response.json()
                
                # Simple pattern matching (could be enhanced with regex)
                matching_entities = []
                for entity in all_entities:
                    entity_id = entity.get("entity_id", "")
                    if pattern in entity_id or entity_id.startswith(pattern.replace("*", "")):
                        matching_entities.append(entity)
                        
                return matching_entities
                
        except aiohttp.ClientError as e:
            raise SensorDataException(f"HTTP error getting entities: {e}")
            
    async def health_check(self) -> bool:
        """Check if Home Assistant is healthy and connected."""
        try:
            if not self.is_connected:
                return False
                
            # Quick API test
            await self._test_api_connection()
            return True
            
        except Exception:
            return False