#!/usr/bin/env python3
"""Development database setup script.

This script provides a simple way to set up the development database
with proper schema and initial data for the occupancy prediction system.
"""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.connection import (
    get_async_session,
    close_database_engine,
)

console = Console()


async def check_database_connection():
    """Check if database is accessible."""
    try:
        settings = Settings()
        async with get_async_session(settings) as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        console.print(f"[red]Database connection failed: {e}[/red]")
        return False


async def check_alembic_status():
    """Check if Alembic is initialized and current."""
    try:
        settings = Settings()
        async with get_async_session(settings) as session:
            # Check if alembic_version table exists
            result = await session.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'alembic_version'
                )
            """)
            has_alembic = result.scalar()
            
            if has_alembic:
                result = await session.execute("SELECT version_num FROM alembic_version")
                version = result.scalar()
                console.print(f"[green]Alembic initialized at version: {version}[/green]")
                return True, version
            else:
                console.print("[yellow]Alembic not initialized[/yellow]")
                return False, None
                
    except Exception as e:
        console.print(f"[red]Error checking Alembic status: {e}[/red]")
        return False, None


async def check_tables_exist():
    """Check if expected tables exist."""
    expected_tables = ["sensor_readings", "room_transitions", "predictions"]
    
    try:
        settings = Settings()
        async with get_async_session(settings) as session:
            result = await session.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            existing_tables = [row[0] for row in result.fetchall()]
            
            missing_tables = set(expected_tables) - set(existing_tables)
            
            if not missing_tables:
                console.print(f"[green]All expected tables exist: {existing_tables}[/green]")
                return True
            else:
                console.print(f"[yellow]Missing tables: {missing_tables}[/yellow]")
                console.print(f"[yellow]Existing tables: {existing_tables}[/yellow]")
                return False
                
    except Exception as e:
        console.print(f"[red]Error checking tables: {e}[/red]")
        return False


def main():
    """Main setup function."""
    console.print("[bold blue]Occupancy Predictor - Database Setup[/bold blue]")
    console.print("=" * 50)
    
    async def setup():
        # Step 1: Check database connection
        console.print("\n[bold]Step 1: Testing database connection[/bold]")
        if not await check_database_connection():
            console.print("\n[red]❌ Setup failed: Cannot connect to database[/red]")
            console.print("\n[yellow]Make sure PostgreSQL is running and credentials are correct.[/yellow]")
            console.print("[yellow]Example Docker command:[/yellow]")
            console.print("""docker run --name occupancy-postgres \\
    -e POSTGRES_PASSWORD=occupancy_password \\
    -e POSTGRES_USER=occupancy \\
    -e POSTGRES_DB=occupancy_dev \\
    -p 5432:5432 \\
    -d postgres:16""")
            return False
        
        console.print("[green]✓ Database connection successful[/green]")
        
        # Step 2: Check Alembic status
        console.print("\n[bold]Step 2: Checking migration status[/bold]")
        alembic_ready, version = await check_alembic_status()
        
        # Step 3: Check tables
        console.print("\n[bold]Step 3: Checking database schema[/bold]")
        tables_exist = await check_tables_exist()
        
        # Step 4: Provide recommendations
        console.print("\n[bold]Step 4: Setup recommendations[/bold]")
        
        if not alembic_ready or not tables_exist:
            console.print("\n[yellow]Database needs initialization. Run:[/yellow]")
            console.print("[cyan]make db-init[/cyan]")
            console.print("or")
            console.print("[cyan]poetry run alembic upgrade head[/cyan]")
        else:
            console.print("\n[green]✅ Database is ready for development![/green]")
            
            # Show additional useful commands
            console.print("\n[bold]Useful commands:[/bold]")
            console.print("[cyan]make db-check[/cyan]        - Check database status")
            console.print("[cyan]make db-migrate[/cyan]      - Run pending migrations")
            console.print("[cyan]make db-rollback[/cyan]     - Rollback last migration")
            console.print("[cyan]make db-reset[/cyan]        - Reset database (⚠️  destroys data)")
            
        await close_database_engine()
        return True
    
    success = asyncio.run(setup())
    
    if success:
        console.print("\n[green]Setup check completed![/green]")
    else:
        console.print("\n[red]Setup check failed![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()