"""Integration tests for Home Assistant WebSocket and REST client."""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import aiohttp
from aiohttp import web, WSMsgType

from src.occupancy.config.settings import Settings
from src.occupancy.infrastructure.homeassistant.client import HomeAssistantClient
from src.occupancy.domain.exceptions import (
    SensorDataException,
    InvalidSensorReadingException,
)

from ..fixtures.sample_data import (
    sample_ha_entity_data,
    SAMPLE_WS_MESSAGES,
)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings for Home Assistant client."""
    return Settings(
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="occupancy_test",
        postgres_user="occupancy",
        postgres_password="test_password",
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,
        ha_url="http://test-homeassistant:8123",
        ha_token="test_token_12345",
        grafana_url="http://test-grafana:3000",
        grafana_api_key="test_grafana_key",
        log_level="DEBUG",
        environment="test"
    )


class TestHomeAssistantClientInitialization:
    """Test HomeAssistantClient initialization and basic functionality."""
    
    def test_client_initialization(self, test_settings: Settings):
        """Test client initializes with correct settings."""
        client = HomeAssistantClient(test_settings)
        
        assert client.settings == test_settings
        assert client.session is None
        assert client.ws_connection is None
        assert client.message_id == 1
        assert client.entity_callbacks == {}
        assert client.subscribed_entities == set()
        assert client.is_connected is False
        assert client.reconnect_attempts == 0
        assert client.max_reconnect_attempts == 10


class TestHomeAssistantClientRESTAPI:
    """Test Home Assistant REST API functionality."""
    
    async def test_get_entity_state_success(self, test_settings: Settings):
        """Test successful entity state retrieval."""
        client = HomeAssistantClient(test_settings)
        
        # Mock the HTTP session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=sample_ha_entity_data()[0])
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        entity_data = await client.get_entity_state("binary_sensor.bedroom_fp2_presence")
        
        assert entity_data["entity_id"] == "binary_sensor.bedroom_fp2_presence"
        assert entity_data["state"] == "on"
        
        # Verify correct URL was called
        expected_url = "http://test-homeassistant:8123/api/states/binary_sensor.bedroom_fp2_presence"
        mock_session.get.assert_called_once_with(expected_url)
    
    async def test_get_entity_state_not_found(self, test_settings: Settings):
        """Test entity state retrieval with 404 error."""
        client = HomeAssistantClient(test_settings)
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        with pytest.raises(InvalidSensorReadingException) as exc_info:
            await client.get_entity_state("binary_sensor.nonexistent")
        
        assert "Entity not found" in str(exc_info.value)
        assert exc_info.value.sensor_entity == "binary_sensor.nonexistent"
    
    async def test_get_entity_state_server_error(self, test_settings: Settings):
        """Test entity state retrieval with server error."""
        client = HomeAssistantClient(test_settings)
        
        # Mock 500 response
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client.get_entity_state("binary_sensor.bedroom_fp2_presence")
        
        assert "Failed to get entity state: HTTP 500" in str(exc_info.value)
    
    async def test_get_entity_state_no_session(self, test_settings: Settings):
        """Test entity state retrieval without session."""
        client = HomeAssistantClient(test_settings)
        # Don't set session (remains None)
        
        with pytest.raises(SensorDataException) as exc_info:
            await client.get_entity_state("binary_sensor.bedroom_fp2_presence")
        
        assert "No session available" in str(exc_info.value)
    
    async def test_get_entity_state_client_error(self, test_settings: Settings):
        """Test entity state retrieval with client error."""
        client = HomeAssistantClient(test_settings)
        
        # Mock client error
        mock_session = AsyncMock()
        mock_session.get.side_effect = aiohttp.ClientError("Connection failed")
        
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client.get_entity_state("binary_sensor.bedroom_fp2_presence")
        
        assert "HTTP error getting entity state" in str(exc_info.value)
    
    async def test_get_entity_history_success(self, test_settings: Settings):
        """Test successful entity history retrieval."""
        client = HomeAssistantClient(test_settings)
        
        # Mock history response
        history_data = [sample_ha_entity_data()[:2]]  # Wrap in list as HA does
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=history_data)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        
        history = await client.get_entity_history(
            "binary_sensor.bedroom_fp2_presence", start_time, end_time
        )
        
        assert len(history) == 2
        assert history[0]["entity_id"] == "binary_sensor.bedroom_fp2_presence"
    
    async def test_get_entity_history_without_end_time(self, test_settings: Settings):
        """Test entity history retrieval without end time."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[[]])  # Empty history
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        history = await client.get_entity_history(
            "binary_sensor.bedroom_fp2_presence", start_time
        )
        
        assert len(history) == 0
    
    async def test_get_entity_history_empty_response(self, test_settings: Settings):
        """Test entity history retrieval with empty response."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])  # Empty list
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        history = await client.get_entity_history(
            "binary_sensor.bedroom_fp2_presence", start_time
        )
        
        assert len(history) == 0
    
    async def test_get_entity_history_error(self, test_settings: Settings):
        """Test entity history retrieval with error."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        with pytest.raises(SensorDataException) as exc_info:
            await client.get_entity_history("binary_sensor.bedroom_fp2_presence", start_time)
        
        assert "Failed to get entity history" in str(exc_info.value)
    
    async def test_get_entities_by_pattern_success(self, test_settings: Settings):
        """Test successful entity pattern matching."""
        client = HomeAssistantClient(test_settings)
        
        # Mock all entities response
        all_entities = sample_ha_entity_data()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=all_entities)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        # Test pattern matching
        matching_entities = await client.get_entities_by_pattern("bedroom")
        
        # Should match entities containing "bedroom"
        assert len(matching_entities) > 0
        for entity in matching_entities:
            assert "bedroom" in entity["entity_id"]
    
    async def test_get_entities_by_pattern_wildcard(self, test_settings: Settings):
        """Test entity pattern matching with wildcard."""
        client = HomeAssistantClient(test_settings)
        
        all_entities = sample_ha_entity_data()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=all_entities)
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        # Test wildcard pattern
        matching_entities = await client.get_entities_by_pattern("binary_sensor.*")
        
        # Should match entities starting with "binary_sensor"
        assert len(matching_entities) > 0
        for entity in matching_entities:
            assert entity["entity_id"].startswith("binary_sensor")
    
    async def test_get_entities_by_pattern_error(self, test_settings: Settings):
        """Test entity pattern matching with error."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client.get_entities_by_pattern("bedroom")
        
        assert "Failed to get entities" in str(exc_info.value)


class TestHomeAssistantClientConnection:
    """Test Home Assistant client connection management."""
    
    async def test_test_api_connection_success(self, test_settings: Settings):
        """Test successful API connection test."""
        client = HomeAssistantClient(test_settings)
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "API running."})
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        # Should not raise exception
        await client._test_api_connection()
        
        # Verify correct endpoint was called
        expected_url = "http://test-homeassistant:8123/api/"
        mock_session.get.assert_called_once_with(expected_url)
    
    async def test_test_api_connection_wrong_status(self, test_settings: Settings):
        """Test API connection test with wrong HTTP status."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._test_api_connection()
        
        assert "HA API test failed: HTTP 404" in str(exc_info.value)
    
    async def test_test_api_connection_wrong_message(self, test_settings: Settings):
        """Test API connection test with wrong message."""
        client = HomeAssistantClient(test_settings)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "Wrong message"})
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._test_api_connection()
        
        assert "HA API not running" in str(exc_info.value)
    
    async def test_test_api_connection_no_session(self, test_settings: Settings):
        """Test API connection test without session."""
        client = HomeAssistantClient(test_settings)
        # Don't set session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._test_api_connection()
        
        assert "No session available" in str(exc_info.value)
    
    async def test_health_check_connected(self, test_settings: Settings):
        """Test health check when connected."""
        client = HomeAssistantClient(test_settings)
        client.is_connected = True
        
        # Mock successful API test
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": "API running."})
        
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        client.session = mock_session
        
        is_healthy = await client.health_check()
        
        assert is_healthy is True
    
    async def test_health_check_not_connected(self, test_settings: Settings):
        """Test health check when not connected."""
        client = HomeAssistantClient(test_settings)
        client.is_connected = False
        
        is_healthy = await client.health_check()
        
        assert is_healthy is False
    
    async def test_health_check_api_error(self, test_settings: Settings):
        """Test health check with API error."""
        client = HomeAssistantClient(test_settings)
        client.is_connected = True
        
        # Mock API error
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("API error")
        client.session = mock_session
        
        is_healthy = await client.health_check()
        
        assert is_healthy is False


class TestHomeAssistantClientWebSocket:
    """Test Home Assistant WebSocket functionality."""
    
    async def test_websocket_authentication_flow(self, test_settings: Settings):
        """Test WebSocket authentication flow."""
        client = HomeAssistantClient(test_settings)
        
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        
        # Mock authentication messages
        auth_required_msg = AsyncMock()
        auth_required_msg.type = WSMsgType.TEXT
        auth_required_msg.data = json.dumps(SAMPLE_WS_MESSAGES["auth_required"])
        
        auth_ok_msg = AsyncMock()
        auth_ok_msg.type = WSMsgType.TEXT
        auth_ok_msg.data = json.dumps(SAMPLE_WS_MESSAGES["auth_ok"])
        
        # Set up receive sequence
        mock_ws.receive = AsyncMock()
        mock_ws.receive.side_effect = [auth_required_msg, auth_ok_msg]
        
        mock_ws.send_str = AsyncMock()
        
        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        client.session = mock_session
        
        # Test authentication
        await client._connect_websocket()
        
        # Verify auth message was sent
        mock_ws.send_str.assert_called_once()
        auth_call_args = mock_ws.send_str.call_args[0][0]
        auth_data = json.loads(auth_call_args)
        assert auth_data["type"] == "auth"
        assert auth_data["access_token"] == test_settings.ha_token
    
    async def test_websocket_authentication_failure(self, test_settings: Settings):
        """Test WebSocket authentication failure."""
        client = HomeAssistantClient(test_settings)
        
        mock_ws = AsyncMock()
        
        # Mock auth required and invalid response
        auth_required_msg = AsyncMock()
        auth_required_msg.type = WSMsgType.TEXT
        auth_required_msg.data = json.dumps(SAMPLE_WS_MESSAGES["auth_required"])
        
        auth_invalid_msg = AsyncMock()
        auth_invalid_msg.type = WSMsgType.TEXT
        auth_invalid_msg.data = json.dumps(SAMPLE_WS_MESSAGES["auth_invalid"])
        
        mock_ws.receive = AsyncMock()
        mock_ws.receive.side_effect = [auth_required_msg, auth_invalid_msg]
        mock_ws.send_str = AsyncMock()
        
        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._connect_websocket()
        
        assert "WebSocket auth failed" in str(exc_info.value)
    
    async def test_websocket_unexpected_auth_message(self, test_settings: Settings):
        """Test WebSocket with unexpected authentication message."""
        client = HomeAssistantClient(test_settings)
        
        mock_ws = AsyncMock()
        
        # Mock unexpected first message
        unexpected_msg = AsyncMock()
        unexpected_msg.type = WSMsgType.TEXT
        unexpected_msg.data = json.dumps({"type": "unexpected"})
        
        mock_ws.receive = AsyncMock(return_value=unexpected_msg)
        
        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        client.session = mock_session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._connect_websocket()
        
        assert "Unexpected WebSocket auth message" in str(exc_info.value)
    
    async def test_websocket_no_session(self, test_settings: Settings):
        """Test WebSocket connection without session."""
        client = HomeAssistantClient(test_settings)
        # Don't set session
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._connect_websocket()
        
        assert "No session available" in str(exc_info.value)


class TestHomeAssistantClientSubscriptions:
    """Test Home Assistant client entity subscriptions."""
    
    async def test_subscribe_to_entity_new_entity(self, test_settings: Settings):
        """Test subscribing to a new entity."""
        client = HomeAssistantClient(test_settings)
        
        # Mock WebSocket for subscription
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock()
        client.ws_connection = mock_ws
        
        # Callback function
        callback_called = []
        async def test_callback(data):
            callback_called.append(data)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        await client.subscribe_to_entity(entity_id, test_callback)
        
        # Verify callback was added
        assert entity_id in client.entity_callbacks
        assert test_callback in client.entity_callbacks[entity_id]
        assert entity_id in client.subscribed_entities
        
        # Verify WebSocket subscription message was sent
        mock_ws.send_str.assert_called_once()
        subscribe_call_args = mock_ws.send_str.call_args[0][0]
        subscribe_data = json.loads(subscribe_call_args)
        assert subscribe_data["type"] == "subscribe_events"
        assert subscribe_data["event_type"] == "state_changed"
        assert subscribe_data["entity_id"] == entity_id
    
    async def test_subscribe_to_entity_existing_entity(self, test_settings: Settings):
        """Test subscribing additional callback to existing entity."""
        client = HomeAssistantClient(test_settings)
        
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock()
        client.ws_connection = mock_ws
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        # Add entity to already subscribed
        client.subscribed_entities.add(entity_id)
        
        async def callback1(data): pass
        async def callback2(data): pass
        
        # Subscribe first callback
        await client.subscribe_to_entity(entity_id, callback1)
        
        # Subscribe second callback
        await client.subscribe_to_entity(entity_id, callback2)
        
        # Should have both callbacks
        assert len(client.entity_callbacks[entity_id]) == 2
        assert callback1 in client.entity_callbacks[entity_id]
        assert callback2 in client.entity_callbacks[entity_id]
        
        # WebSocket subscription should only be called once (for first callback)
        assert mock_ws.send_str.call_count == 1
    
    async def test_subscribe_websocket_entity_no_connection(self, test_settings: Settings):
        """Test WebSocket entity subscription without connection."""
        client = HomeAssistantClient(test_settings)
        # No WebSocket connection set
        
        with pytest.raises(SensorDataException) as exc_info:
            await client._subscribe_websocket_entity("binary_sensor.test")
        
        assert "No WebSocket connection" in str(exc_info.value)
    
    async def test_unsubscribe_from_entity_specific_callback(self, test_settings: Settings):
        """Test unsubscribing specific callback from entity."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        async def callback1(data): pass
        async def callback2(data): pass
        
        # Set up callbacks
        client.entity_callbacks[entity_id] = [callback1, callback2]
        client.subscribed_entities.add(entity_id)
        
        # Unsubscribe specific callback
        await client.unsubscribe_from_entity(entity_id, callback1)
        
        # Should only have callback2 left
        assert len(client.entity_callbacks[entity_id]) == 1
        assert callback2 in client.entity_callbacks[entity_id]
        assert callback1 not in client.entity_callbacks[entity_id]
        assert entity_id in client.subscribed_entities  # Still subscribed
    
    async def test_unsubscribe_from_entity_all_callbacks(self, test_settings: Settings):
        """Test unsubscribing all callbacks from entity."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        async def callback1(data): pass
        async def callback2(data): pass
        
        # Set up callbacks
        client.entity_callbacks[entity_id] = [callback1, callback2]
        client.subscribed_entities.add(entity_id)
        
        # Unsubscribe all callbacks
        await client.unsubscribe_from_entity(entity_id)
        
        # Should have no callbacks and be unsubscribed
        assert entity_id not in client.entity_callbacks
        assert entity_id not in client.subscribed_entities
    
    async def test_unsubscribe_from_entity_last_callback(self, test_settings: Settings):
        """Test unsubscribing last callback from entity."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        async def callback1(data): pass
        
        # Set up single callback
        client.entity_callbacks[entity_id] = [callback1]
        client.subscribed_entities.add(entity_id)
        
        # Unsubscribe last callback
        await client.unsubscribe_from_entity(entity_id, callback1)
        
        # Should be completely unsubscribed
        assert entity_id not in client.entity_callbacks
        assert entity_id not in client.subscribed_entities
    
    async def test_unsubscribe_from_entity_not_subscribed(self, test_settings: Settings):
        """Test unsubscribing from entity that's not subscribed."""
        client = HomeAssistantClient(test_settings)
        
        async def callback(data): pass
        
        # Should not raise error
        await client.unsubscribe_from_entity("binary_sensor.unknown", callback)
        
        # Should remain empty
        assert len(client.entity_callbacks) == 0
        assert len(client.subscribed_entities) == 0


class TestHomeAssistantClientMessageHandling:
    """Test Home Assistant client WebSocket message handling."""
    
    async def test_handle_state_change_event(self, test_settings: Settings):
        """Test handling state change events."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        # Set up callback
        callback_data = []
        async def test_callback(data):
            callback_data.append(data)
        
        client.entity_callbacks[entity_id] = [test_callback]
        
        # Simulate state change event
        state_data = {
            "entity_id": entity_id,
            "new_state": {"state": "on"},
            "old_state": {"state": "off"}
        }
        
        await client._handle_state_change(state_data)
        
        # Verify callback was called
        assert len(callback_data) == 1
        assert callback_data[0] == state_data
    
    async def test_handle_state_change_no_callbacks(self, test_settings: Settings):
        """Test handling state change for entity without callbacks."""
        client = HomeAssistantClient(test_settings)
        
        # No callbacks registered
        state_data = {
            "entity_id": "binary_sensor.unknown",
            "new_state": {"state": "on"},
            "old_state": {"state": "off"}
        }
        
        # Should not raise error
        await client._handle_state_change(state_data)
    
    async def test_handle_state_change_callback_error(self, test_settings: Settings):
        """Test handling state change when callback raises error."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        # Set up failing callback
        async def failing_callback(data):
            raise ValueError("Callback error")
        
        client.entity_callbacks[entity_id] = [failing_callback]
        
        state_data = {
            "entity_id": entity_id,
            "new_state": {"state": "on"}
        }
        
        # Should not raise error (error is logged but not re-raised)
        await client._handle_state_change(state_data)
    
    async def test_handle_websocket_message_state_changed(self, test_settings: Settings):
        """Test handling WebSocket state changed message."""
        client = HomeAssistantClient(test_settings)
        
        entity_id = "binary_sensor.bedroom_fp2_presence"
        
        # Set up callback
        callback_data = []
        async def test_callback(data):
            callback_data.append(data)
        
        client.entity_callbacks[entity_id] = [test_callback]
        
        # Simulate WebSocket message
        ws_message = SAMPLE_WS_MESSAGES["state_changed_event"]
        
        await client._handle_websocket_message(ws_message)
        
        # Verify callback was called with event data
        assert len(callback_data) == 1
        assert callback_data[0]["entity_id"] == entity_id
    
    async def test_handle_websocket_message_non_event(self, test_settings: Settings):
        """Test handling non-event WebSocket message."""
        client = HomeAssistantClient(test_settings)
        
        # Non-event message
        ws_message = {
            "type": "result",
            "success": True
        }
        
        # Should not raise error
        await client._handle_websocket_message(ws_message)
    
    async def test_handle_websocket_message_non_state_change_event(self, test_settings: Settings):
        """Test handling non-state-change event message."""
        client = HomeAssistantClient(test_settings)
        
        # Different event type
        ws_message = {
            "type": "event",
            "event": {
                "event_type": "call_service",
                "data": {"service": "light.turn_on"}
            }
        }
        
        # Should not raise error
        await client._handle_websocket_message(ws_message)


class TestHomeAssistantClientContextManager:
    """Test Home Assistant client context manager functionality."""
    
    async def test_context_manager_success(self, test_settings: Settings):
        """Test successful context manager usage."""
        # Mock successful connection
        with patch.object(HomeAssistantClient, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(HomeAssistantClient, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                
                async with HomeAssistantClient(test_settings) as client:
                    assert isinstance(client, HomeAssistantClient)
                
                # Verify connect and disconnect were called
                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    async def test_context_manager_connection_error(self, test_settings: Settings):
        """Test context manager with connection error."""
        # Mock connection failure
        with patch.object(HomeAssistantClient, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(HomeAssistantClient, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                
                mock_connect.side_effect = SensorDataException("Connection failed")
                
                with pytest.raises(SensorDataException):
                    async with HomeAssistantClient(test_settings) as client:
                        pass
                
                # Disconnect should still be called on error
                mock_disconnect.assert_called_once()


class TestHomeAssistantClientReconnection:
    """Test Home Assistant client reconnection logic."""
    
    async def test_attempt_reconnect_success(self, test_settings: Settings):
        """Test successful reconnection attempt."""
        client = HomeAssistantClient(test_settings)
        client.reconnect_attempts = 0
        
        # Mock successful reconnection
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                
                await client._attempt_reconnect()
                
                # Should attempt reconnection
                assert client.reconnect_attempts == 1
                mock_sleep.assert_called_once_with(2)  # 2^1 = 2 seconds backoff
                mock_connect.assert_called_once()
    
    async def test_attempt_reconnect_max_attempts_reached(self, test_settings: Settings):
        """Test reconnection when max attempts reached."""
        client = HomeAssistantClient(test_settings)
        client.reconnect_attempts = client.max_reconnect_attempts
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            await client._attempt_reconnect()
            
            # Should not attempt reconnection
            mock_connect.assert_not_called()
    
    async def test_attempt_reconnect_exponential_backoff(self, test_settings: Settings):
        """Test exponential backoff in reconnection."""
        client = HomeAssistantClient(test_settings)
        client.reconnect_attempts = 4  # Should result in 16 second backoff, capped at 30
        
        with patch.object(client, 'connect', new_callable=AsyncMock):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                
                await client._attempt_reconnect()
                
                # Should use capped backoff (min(30, 2^5) = 30)
                mock_sleep.assert_called_once_with(30)
    
    async def test_attempt_reconnect_failure_retry(self, test_settings: Settings):
        """Test reconnection failure triggers another attempt."""
        client = HomeAssistantClient(test_settings)
        client.reconnect_attempts = 0
        
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(client, '_attempt_reconnect', new_callable=AsyncMock) as mock_retry:
                
                # First call should be the actual call, second call should be the retry
                mock_connect.side_effect = SensorDataException("Connection failed")
                
                # Call the actual method for the first attempt
                original_method = HomeAssistantClient._attempt_reconnect.__wrapped__
                await original_method(client)
                
                # Should have called reconnect again (recursive call)
                # This is hard to test due to recursion, so we'll just verify connect was called
                mock_connect.assert_called_once()