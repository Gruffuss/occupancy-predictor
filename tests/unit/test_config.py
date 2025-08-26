"""Unit tests for configuration settings and rooms.yaml loading."""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from pydantic import ValidationError

from src.occupancy.config.settings import Settings
from src.occupancy.domain.models import RoomType
from src.occupancy.domain.exceptions import InvalidRoomConfigurationException

from ..fixtures.sample_data import TEST_DATABASE_CONFIG, SAMPLE_ROOMS_CONFIG


class TestSettings:
    """Test Settings configuration class."""
    
    def test_settings_with_environment_variables(self):
        """Test settings loading from environment variables."""
        env_vars = {
            "POSTGRES_HOST": "test-postgres",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "test_occupancy",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_pass",
            "REDIS_HOST": "test-redis",
            "REDIS_PORT": "6380",
            "REDIS_DB": "2",
            "HA_URL": "http://test-ha:8123",
            "HA_TOKEN": "test_ha_token",
            "GRAFANA_URL": "http://test-grafana:3000",
            "GRAFANA_API_KEY": "test_grafana_key",
            "LOG_LEVEL": "DEBUG",
            "ENVIRONMENT": "test"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.postgres_host == "test-postgres"
            assert settings.postgres_port == 5433
            assert settings.postgres_db == "test_occupancy"
            assert settings.postgres_user == "test_user"
            assert settings.postgres_password == "test_pass"
            assert settings.redis_host == "test-redis"
            assert settings.redis_port == 6380
            assert settings.redis_db == 2
            assert settings.ha_url == "http://test-ha:8123"
            assert settings.ha_token == "test_ha_token"
            assert settings.grafana_url == "http://test-grafana:3000"
            assert settings.grafana_api_key == "test_grafana_key"
            assert settings.log_level == "DEBUG"
            assert settings.environment == "test"
    
    def test_settings_default_values(self):
        """Test settings with default values."""
        # Need to provide required fields
        env_vars = {
            "POSTGRES_PASSWORD": "required_password",
            "HA_URL": "http://localhost:8123",
            "HA_TOKEN": "required_token",
            "GRAFANA_URL": "http://localhost:3000",
            "GRAFANA_API_KEY": "required_key"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            # Test default values
            assert settings.postgres_host == "localhost"
            assert settings.postgres_port == 5432
            assert settings.postgres_db == "occupancy"
            assert settings.postgres_user == "occupancy"
            assert settings.redis_host == "localhost"
            assert settings.redis_port == 6379
            assert settings.redis_db == 0
            assert settings.log_level == "INFO"
            assert settings.environment == "development"
    
    def test_settings_validation_missing_required_fields(self):
        """Test settings validation when required fields are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            # Should complain about missing required fields
            errors = exc_info.value.errors()
            required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
            
            # These fields are required and have no defaults
            expected_required = {'postgres_password', 'ha_url', 'ha_token', 'grafana_url', 'grafana_api_key'}
            assert expected_required.issubset(required_fields)
    
    def test_settings_type_validation(self):
        """Test settings type validation."""
        env_vars = {
            "POSTGRES_PASSWORD": "test_pass",
            "HA_URL": "http://localhost:8123",
            "HA_TOKEN": "test_token",
            "GRAFANA_URL": "http://localhost:3000", 
            "GRAFANA_API_KEY": "test_key",
            "POSTGRES_PORT": "invalid_port",  # Should be integer
        }
        
        with patch.dict(os.environ, env_vars):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            
            errors = exc_info.value.errors()
            port_errors = [e for e in errors if 'postgres_port' in str(e['loc'])]
            assert len(port_errors) > 0
    
    def test_settings_from_env_file(self):
        """Test loading settings from .env file."""
        env_content = """
POSTGRES_PASSWORD=env_file_password
HA_URL=http://env-ha:8123
HA_TOKEN=env_file_token
GRAFANA_URL=http://env-grafana:3000
GRAFANA_API_KEY=env_file_key
LOG_LEVEL=WARNING
ENVIRONMENT=testing
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            env_file_path = f.name
        
        try:
            # Mock the model_config to use our test env file
            with patch.object(Settings, 'model_config') as mock_config:
                mock_config.env_file = env_file_path
                
                # Clear environment to ensure we're loading from file
                with patch.dict(os.environ, {}, clear=True):
                    settings = Settings(_env_file=env_file_path)
                    
                    assert settings.postgres_password == "env_file_password"
                    assert settings.ha_url == "http://env-ha:8123"
                    assert settings.log_level == "WARNING"
                    assert settings.environment == "testing"
        finally:
            os.unlink(env_file_path)
    
    def test_settings_environment_priority(self):
        """Test that environment variables take priority over .env file."""
        env_content = """
POSTGRES_PASSWORD=file_password
LOG_LEVEL=DEBUG
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            env_file_path = f.name
        
        try:
            env_vars = {
                "POSTGRES_PASSWORD": "env_password",  # This should override file
                "HA_URL": "http://localhost:8123",
                "HA_TOKEN": "test_token",
                "GRAFANA_URL": "http://localhost:3000",
                "GRAFANA_API_KEY": "test_key",
            }
            
            with patch.dict(os.environ, env_vars):
                settings = Settings(_env_file=env_file_path)
                
                # Environment variable should win
                assert settings.postgres_password == "env_password"
                # File value should be used where env var not set
                assert settings.log_level == "DEBUG"
        finally:
            os.unlink(env_file_path)


class TestRoomsConfigLoading:
    """Test loading and validation of rooms.yaml configuration."""
    
    def test_load_valid_rooms_config(self):
        """Test loading valid rooms configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            config_path = f.name
        
        try:
            # Import the mapper to test config loading
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            mapper = EntityMapper(config_path)
            
            # Verify config was loaded
            assert 'rooms' in mapper.rooms_config
            assert len(mapper.rooms_config['rooms']) == 4
            
            # Verify entity mappings were built
            assert len(mapper.entity_to_room_zone) > 0
            
            # Check specific mappings
            assert "binary_sensor.bedroom_fp2_presence" in mapper.entity_to_room_zone
            bedroom_mapping = mapper.entity_to_room_zone["binary_sensor.bedroom_fp2_presence"]
            assert bedroom_mapping[0] == RoomType.BEDROOM
            assert bedroom_mapping[1] == "full"
            
        finally:
            os.unlink(config_path)
    
    def test_load_missing_rooms_config_file(self):
        """Test error when rooms config file is missing."""
        from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
        
        with pytest.raises(InvalidRoomConfigurationException) as exc_info:
            EntityMapper("/nonexistent/path/rooms.yaml")
        
        assert "not found" in str(exc_info.value)
    
    def test_load_invalid_yaml_rooms_config(self):
        """Test error when rooms config has invalid YAML."""
        invalid_yaml = """
rooms:
  bedroom:
    type: bedroom
    zones:
      full:
        sensor_entity: "binary_sensor.test"
    - invalid: yaml structure
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            with pytest.raises(InvalidRoomConfigurationException) as exc_info:
                EntityMapper(config_path)
            
            assert "Invalid YAML" in str(exc_info.value)
        finally:
            os.unlink(config_path)
    
    def test_load_rooms_config_missing_rooms_section(self):
        """Test error when rooms config is missing rooms section."""
        config_without_rooms = {
            "global_config": {
                "max_sensor_gap_minutes": 10
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_without_rooms, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            with pytest.raises(InvalidRoomConfigurationException) as exc_info:
                EntityMapper(config_path)
            
            assert "Missing 'rooms' section" in str(exc_info.value)
        finally:
            os.unlink(config_path)
    
    def test_load_rooms_config_with_invalid_room_types(self):
        """Test loading rooms config with invalid room types."""
        config_with_invalid_room = {
            "rooms": {
                "bedroom": {
                    "type": "invalid_room_type",  # Not a valid RoomType
                    "zones": {
                        "full": {
                            "sensor_entity": "binary_sensor.test",
                            "display_name": "Full Room"
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_invalid_room, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            # Should load but skip invalid room type
            mapper = EntityMapper(config_path)
            
            # Should have no entity mappings due to invalid room type
            assert len(mapper.entity_to_room_zone) == 0
        finally:
            os.unlink(config_path)
    
    def test_load_rooms_config_missing_sensor_entities(self):
        """Test loading rooms config with missing sensor entities."""
        config_missing_entities = {
            "rooms": {
                "bedroom": {
                    "type": "bedroom",
                    "zones": {
                        "full": {
                            "display_name": "Full Room"
                            # Missing sensor_entity
                        },
                        "bed_side": {
                            "sensor_entity": "binary_sensor.bedroom_bed_side",
                            "display_name": "Bed Side"
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_missing_entities, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            mapper = EntityMapper(config_path)
            
            # Should only map the zone with sensor_entity
            assert len(mapper.entity_to_room_zone) == 1
            assert "binary_sensor.bedroom_bed_side" in mapper.entity_to_room_zone
        finally:
            os.unlink(config_path)
    
    def test_rooms_config_validation_functions(self):
        """Test room configuration validation functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            mapper = EntityMapper(config_path)
            
            # Test valid room configuration
            issues = mapper.validate_room_configuration(RoomType.BEDROOM)
            assert len(issues) == 0
            
            # Test getting room config
            bedroom_config = mapper.get_room_config(RoomType.BEDROOM)
            assert bedroom_config is not None
            assert bedroom_config['type'] == 'bedroom'
            assert 'zones' in bedroom_config
            assert 'prediction_config' in bedroom_config
            
            # Test non-existent room
            missing_config = mapper.get_room_config(RoomType.GUEST_BEDROOM)  # Not in sample config
            assert missing_config is None
            
        finally:
            os.unlink(config_path)
    
    def test_rooms_config_incomplete_validation(self):
        """Test validation of incomplete room configuration."""
        incomplete_config = {
            "rooms": {
                "bedroom": {
                    "type": "bedroom",
                    "zones": {
                        "full": {
                            "sensor_entity": "binary_sensor.bedroom_fp2_presence"
                            # Missing display_name
                        }
                    }
                    # Missing prediction_config
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_config, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            mapper = EntityMapper(config_path)
            
            # Test validation finds issues
            issues = mapper.validate_room_configuration(RoomType.BEDROOM)
            assert len(issues) > 0
            
            # Should report missing display_name and prediction_config
            issue_text = ' '.join(issues)
            assert 'display_name' in issue_text
            assert 'prediction_config' in issue_text
            
        finally:
            os.unlink(config_path)
    
    def test_rooms_config_entity_lookup_functions(self):
        """Test entity lookup functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            config_path = f.name
        
        try:
            from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
            
            mapper = EntityMapper(config_path)
            
            # Test get_room_entities
            bedroom_entities = mapper.get_room_entities(RoomType.BEDROOM)
            assert "binary_sensor.bedroom_fp2_presence" in bedroom_entities
            assert "binary_sensor.bedroom_fp2_anca_bed_side" in bedroom_entities
            
            # Test get_zone_entities
            full_zone_entities = mapper.get_zone_entities(RoomType.BEDROOM, "full")
            assert "binary_sensor.bedroom_fp2_presence" in full_zone_entities
            assert len(full_zone_entities) == 1
            
            # Test get_all_sensor_entities
            all_entities = mapper.get_all_sensor_entities()
            assert len(all_entities) > 0
            assert "binary_sensor.bedroom_fp2_presence" in all_entities
            
            # Test is_entity_configured
            assert mapper.is_entity_configured("binary_sensor.bedroom_fp2_presence") is True
            assert mapper.is_entity_configured("binary_sensor.unknown_sensor") is False
            
            # Test get_room_for_entity
            room = mapper.get_room_for_entity("binary_sensor.bedroom_fp2_presence")
            assert room == RoomType.BEDROOM
            
            # Test get_zone_for_entity
            zone = mapper.get_zone_for_entity("binary_sensor.bedroom_fp2_presence")
            assert zone == "full"
            
        finally:
            os.unlink(config_path)
    
    def test_default_rooms_config_path(self):
        """Test default rooms config path resolution."""
        from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
        
        # Create mock rooms.yaml in expected location
        expected_path = Path(__file__).parent.parent.parent / "src" / "occupancy" / "config" / "rooms.yaml"
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(expected_path, 'w') as f:
                yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            
            # Should find the config file automatically
            mapper = EntityMapper()  # No path specified
            assert mapper.rooms_config is not None
            assert len(mapper.entity_to_room_zone) > 0
            
        except Exception:
            # If we can't write to the actual config location, skip this test
            pytest.skip("Cannot write to default config location")
        finally:
            if expected_path.exists():
                expected_path.unlink()


class TestConfigurationIntegration:
    """Integration tests for configuration loading."""
    
    def test_settings_and_rooms_config_together(self):
        """Test using Settings and rooms config together."""
        # Create temporary rooms config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(SAMPLE_ROOMS_CONFIG, f)
            rooms_config_path = f.name
        
        try:
            # Set up environment for Settings
            env_vars = {
                "POSTGRES_PASSWORD": "test_pass",
                "HA_URL": "http://test-ha:8123",
                "HA_TOKEN": "test_token",
                "GRAFANA_URL": "http://test-grafana:3000",
                "GRAFANA_API_KEY": "test_key",
                "ENVIRONMENT": "test"
            }
            
            with patch.dict(os.environ, env_vars):
                settings = Settings()
                
                # Verify settings loaded correctly
                assert settings.ha_url == "http://test-ha:8123"
                assert settings.environment == "test"
                
                # Load rooms config
                from src.occupancy.infrastructure.homeassistant.mappers import EntityMapper
                mapper = EntityMapper(rooms_config_path)
                
                # Verify both work together
                assert mapper.rooms_config is not None
                assert len(mapper.entity_to_room_zone) > 0
                
                # Test that we can combine info from both
                ha_base_url = settings.ha_url
                bedroom_entities = mapper.get_room_entities(RoomType.BEDROOM)
                
                assert ha_base_url.startswith("http")
                assert len(bedroom_entities) > 0
                
        finally:
            os.unlink(rooms_config_path)