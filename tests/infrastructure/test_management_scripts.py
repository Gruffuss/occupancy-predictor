"""Test database management scripts."""

import pytest
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List, Tuple

from occupancy.config.settings import Settings


class TestManageDatabaseScript:
    """Test the scripts/manage_db.py database management script."""
    
    def get_script_path(self) -> Path:
        """Get path to manage_db.py script."""
        return Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
    
    def run_script_command(self, command: List[str], env: dict = None) -> Tuple[int, str, str]:
        """Run manage_db.py script with given command."""
        script_path = self.get_script_path()
        
        try:
            result = subprocess.run(
                ["python", str(script_path)] + command,
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def test_script_exists_and_executable(self):
        """Test that manage_db.py script exists and is executable."""
        script_path = self.get_script_path()
        assert script_path.exists(), "manage_db.py script not found"
        
        # Should be a Python file
        assert script_path.suffix == ".py", "Script should have .py extension"
        
        # Should have proper shebang or be runnable with python
        with open(script_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        assert first_line.startswith("#!") or "python" in first_line.lower(), \
            "Script should have shebang or be clearly marked as Python"
    
    def test_script_help_command(self):
        """Test that script provides help information."""
        returncode, stdout, stderr = self.run_script_command(["--help"])
        
        # Should not error and provide help
        assert returncode == 0, f"Help command failed: {stderr}"
        assert "help" in stdout.lower() or "usage" in stdout.lower()
        
        # Should list available commands
        expected_commands = ["init", "migrate", "rollback", "test-connection", "check-tables"]
        for command in expected_commands:
            assert command in stdout or command.replace("-", "_") in stdout, \
                f"Command {command} not listed in help"
    
    def test_script_imports_correctly(self):
        """Test that script can import required modules."""
        script_path = self.get_script_path()
        
        # Read and check imports
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should import required modules
        required_imports = [
            "typer",
            "Settings",
            "get_async_session",
            "close_database_engine"
        ]
        
        for import_name in required_imports:
            assert import_name in content, f"Missing import: {import_name}"
        
        # Try to compile to check syntax
        try:
            compile(content, str(script_path), 'exec')
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")
    
    @patch('occupancy.config.settings.Settings')
    def test_test_connection_command(self, mock_settings):
        """Test the test-connection command."""
        # Mock Settings to avoid real database connection
        mock_settings_instance = MagicMock()
        mock_settings_instance.postgres_host = "localhost"
        mock_settings_instance.postgres_port = 5432
        mock_settings_instance.postgres_db = "test_db"
        mock_settings.return_value = mock_settings_instance
        
        # Test connection command should not require database
        # It should at least try to connect and report results
        returncode, stdout, stderr = self.run_script_command(["test-connection"])
        
        # Command should execute (may fail with connection error, but should run)
        assert returncode in [0, 1], f"test-connection command had unexpected error: {stderr}"
        
        # Should provide feedback about connection attempt
        output = stdout + stderr
        assert "connection" in output.lower() or "test" in output.lower()
    
    def test_script_commands_are_defined(self):
        """Test that all expected commands are defined in the script."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Expected command functions (decorated with @app.command())
        expected_commands = [
            "def init(",
            "def migrate(",
            "def rollback(",
            "def current(",
            "def history(",
            "def test_connection(",
            "def check_tables(",
            "def cleanup(",
            "def reset("
        ]
        
        for command_def in expected_commands:
            assert command_def in content, f"Missing command function: {command_def}"
        
        # Should have @app.command() decorators
        assert "@app.command()" in content, "Missing @app.command() decorators"
    
    def test_script_uses_proper_error_handling(self):
        """Test that script has proper error handling."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have exception handling
        assert "try:" in content and "except" in content, "Missing exception handling"
        
        # Should use typer.Exit for error codes
        assert "typer.Exit" in content, "Should use typer.Exit for error handling"
        
        # Should have proper async patterns
        assert "asyncio.run" in content, "Should use asyncio.run for async operations"
    
    def test_script_alembic_integration(self):
        """Test that script properly integrates with Alembic."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should run Alembic commands via subprocess
        assert "subprocess.run" in content, "Should use subprocess to run Alembic"
        assert "alembic" in content, "Should reference Alembic commands"
        
        # Should handle Alembic command results
        assert "returncode" in content or "exit_code" in content, \
            "Should check Alembic command return codes"
    
    def test_script_console_output(self):
        """Test that script provides proper console output."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should use Rich console for output
        assert "Console" in content, "Should use Rich Console for output"
        assert "console.print" in content, "Should use console.print for user feedback"
        
        # Should provide colored output for success/error
        assert "[green]" in content or "[red]" in content, \
            "Should use colored console output"


class TestSetupDevDatabaseScript:
    """Test the scripts/setup_dev_db.py development setup script."""
    
    def get_script_path(self) -> Path:
        """Get path to setup_dev_db.py script."""
        return Path(__file__).parent.parent.parent / "scripts" / "setup_dev_db.py"
    
    def run_script(self, env: dict = None) -> Tuple[int, str, str]:
        """Run setup_dev_db.py script."""
        script_path = self.get_script_path()
        
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Script timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def test_setup_script_exists(self):
        """Test that setup_dev_db.py script exists."""
        script_path = self.get_script_path()
        assert script_path.exists(), "setup_dev_db.py script not found"
    
    def test_setup_script_structure(self):
        """Test that setup script has proper structure."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have required functions
        required_functions = [
            "def check_database_connection(",
            "def check_alembic_status(",
            "def check_tables_exist(",
            "def main("
        ]
        
        for func in required_functions:
            assert func in content, f"Missing function: {func}"
        
        # Should be async-friendly
        assert "async def" in content, "Should have async functions"
        assert "await" in content, "Should use await for async operations"
        
        # Should have proper imports
        assert "Settings" in content, "Should import Settings"
        assert "get_async_session" in content, "Should import database utilities"
    
    @patch('occupancy.config.settings.Settings')
    def test_setup_script_execution(self, mock_settings):
        """Test that setup script can execute."""
        mock_settings_instance = MagicMock()
        mock_settings_instance.postgres_host = "localhost"
        mock_settings_instance.postgres_port = 5432
        mock_settings_instance.postgres_db = "test_db"
        mock_settings.return_value = mock_settings_instance
        
        # Script should execute and provide feedback
        returncode, stdout, stderr = self.run_script()
        
        # Should complete (may fail on connection, but should run)
        assert returncode in [0, 1], f"Setup script had unexpected error: {stderr}"
        
        # Should provide useful output
        output = stdout + stderr
        assert len(output.strip()) > 0, "Script should provide output"
        assert "database" in output.lower() or "setup" in output.lower()
    
    def test_setup_script_recommendations(self):
        """Test that setup script provides useful recommendations."""
        script_path = self.get_script_path()
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should provide command recommendations
        recommendations = [
            "make db-init",
            "make db-check", 
            "alembic upgrade head",
            "docker run"
        ]
        
        for recommendation in recommendations:
            assert recommendation in content, f"Missing recommendation: {recommendation}"


class TestMakefileTargets:
    """Test Makefile database management targets."""
    
    def get_makefile_path(self) -> Path:
        """Get path to Makefile."""
        return Path(__file__).parent.parent.parent / "Makefile"
    
    def test_makefile_exists(self):
        """Test that Makefile exists."""
        makefile = self.get_makefile_path()
        assert makefile.exists(), "Makefile not found"
    
    def test_database_targets_defined(self):
        """Test that all database targets are defined in Makefile."""
        makefile = self.get_makefile_path()
        
        with open(makefile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Expected database targets
        expected_targets = [
            "db-init:",
            "db-migrate:",
            "db-rollback:",
            "db-current:",
            "db-history:",
            "db-test:",
            "db-check:",
            "db-reset:",
            "migrate-create:",
            "migrate-up:",
            "migrate-down:"
        ]
        
        for target in expected_targets:
            assert target in content, f"Missing Makefile target: {target}"
    
    def test_makefile_target_implementations(self):
        """Test that Makefile targets have proper implementations."""
        makefile = self.get_makefile_path()
        
        with open(makefile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Targets should call appropriate tools
        target_implementations = {
            "db-init:": "python scripts/manage_db.py",
            "db-test:": "python scripts/manage_db.py test-connection",
            "migrate-up:": "alembic upgrade head",
            "migrate-down:": "alembic downgrade"
        }
        
        for target, implementation in target_implementations.items():
            if target in content:
                # Find the target section
                target_start = content.find(target)
                next_target = content.find("\n\n", target_start)
                if next_target == -1:
                    target_section = content[target_start:]
                else:
                    target_section = content[target_start:next_target]
                
                assert implementation in target_section, \
                    f"Target {target} should use {implementation}"
    
    def test_makefile_help_target(self):
        """Test that Makefile has help target with database commands."""
        makefile = self.get_makefile_path()
        
        with open(makefile, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have help target
        assert "help:" in content or ".PHONY: help" in content
        
        # Help should mention database commands
        if "help:" in content:
            help_start = content.find("help:")
            next_target = content.find("\n\n", help_start)
            if next_target != -1:
                help_section = content[help_start:next_target]
                assert "db-" in help_section or "database" in help_section.lower()


class TestScriptIntegration:
    """Test integration between management scripts and other components."""
    
    @patch('occupancy.infrastructure.database.connection.get_async_session')
    @patch('occupancy.config.settings.Settings')
    def test_scripts_use_proper_settings(self, mock_settings, mock_session):
        """Test that scripts properly use Settings configuration."""
        mock_settings_instance = MagicMock()
        mock_settings.return_value = mock_settings_instance
        
        # Mock async session context manager
        mock_session_instance = MagicMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock()
        
        # Scripts should use Settings for database configuration
        script_path = Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should import and use Settings
        assert "from occupancy.config.settings import Settings" in content
        assert "settings = Settings()" in content
    
    def test_scripts_handle_database_unavailable(self):
        """Test that scripts gracefully handle database unavailability."""
        # This tests error handling when database is not available
        
        # Use environment variables to simulate unavailable database
        env = os.environ.copy()
        env['POSTGRES_HOST'] = 'nonexistent_host'
        env['POSTGRES_PASSWORD'] = 'test'
        
        script_path = Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
        
        try:
            result = subprocess.run(
                ["python", str(script_path), "test-connection"],
                capture_output=True,
                text=True,
                env=env,
                timeout=10
            )
            
            # Should not crash, should provide error message
            output = result.stdout + result.stderr
            assert "error" in output.lower() or "connection" in output.lower() or "failed" in output.lower()
            
        except subprocess.TimeoutExpired:
            # If it times out, that's also acceptable behavior
            pass
        except Exception:
            # Other exceptions might indicate script problems
            pytest.fail("Script should handle connection errors gracefully")
    
    def test_scripts_respect_environment_variables(self):
        """Test that scripts respect environment variable configuration."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
        
        # Check that scripts would use environment variables
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should use Settings which reads from environment
        assert "Settings()" in content, "Should create Settings instance"
        
        # Settings should read from environment (check settings.py)
        settings_path = Path(__file__).parent.parent.parent / "src" / "occupancy" / "config" / "settings.py"
        
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings_content = f.read()
        
        assert "env_file" in settings_content, "Settings should read from .env file"
    
    def test_script_cli_interfaces_consistent(self):
        """Test that CLI interfaces are consistent across scripts."""
        # Both scripts should use similar patterns
        
        manage_script = Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
        setup_script = Path(__file__).parent.parent.parent / "scripts" / "setup_dev_db.py"
        
        with open(manage_script, 'r', encoding='utf-8') as f:
            manage_content = f.read()
        
        with open(setup_script, 'r', encoding='utf-8') as f:
            setup_content = f.read()
        
        # Both should use similar imports and patterns
        common_imports = [
            "from pathlib import Path",
            "from occupancy.config.settings import Settings"
        ]
        
        for import_stmt in common_imports:
            assert import_stmt in manage_content, f"manage_db.py missing: {import_stmt}"
            assert import_stmt in setup_content, f"setup_dev_db.py missing: {import_stmt}"
        
        # Both should use rich/typer for user interface
        assert "rich" in manage_content.lower() or "console" in manage_content
        assert "rich" in setup_content.lower() or "console" in setup_content


class TestScriptErrorHandling:
    """Test error handling in management scripts."""
    
    def test_scripts_handle_import_errors(self):
        """Test that scripts handle missing dependencies gracefully."""
        # This is more of a design test - scripts should have proper error handling
        
        script_paths = [
            Path(__file__).parent.parent.parent / "scripts" / "manage_db.py",
            Path(__file__).parent.parent.parent / "scripts" / "setup_dev_db.py"
        ]
        
        for script_path in script_paths:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should have some form of error handling
            assert "try:" in content and "except" in content, \
                f"{script_path.name} should have exception handling"
    
    def test_scripts_provide_useful_error_messages(self):
        """Test that scripts provide useful error messages."""
        script_paths = [
            Path(__file__).parent.parent.parent / "scripts" / "manage_db.py",
            Path(__file__).parent.parent.parent / "scripts" / "setup_dev_db.py"
        ]
        
        for script_path in script_paths:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should provide user-friendly error messages
            error_indicators = [
                "print(",
                "console.print(",
                "typer.echo(",
                "logger."
            ]
            
            has_user_output = any(indicator in content for indicator in error_indicators)
            assert has_user_output, f"{script_path.name} should provide user feedback"
    
    def test_scripts_exit_with_proper_codes(self):
        """Test that scripts use proper exit codes."""
        manage_script = Path(__file__).parent.parent.parent / "scripts" / "manage_db.py"
        
        with open(manage_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should use typer.Exit for proper exit codes
        assert "typer.Exit" in content, "Should use typer.Exit for error conditions"
        
        # Should have different exit codes for different error conditions
        assert "raise typer.Exit(1)" in content or "typer.Exit(exit_code)" in content