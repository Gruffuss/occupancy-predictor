#!/usr/bin/env python3
"""Database management utility for occupancy predictor.

This script provides convenient commands for managing database migrations,
creating/dropping tables, and other database operations.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from occupancy.config.settings import Settings
from occupancy.infrastructure.database.connection import (
    close_database_engine,
    create_all_tables,
    drop_all_tables,
    get_async_session,
    get_database_engine,
)

app = typer.Typer(help="Database management utility for occupancy predictor")
console = Console()


def run_alembic_command(command: str) -> int:
    """Run an Alembic command and return exit code."""
    import subprocess
    
    try:
        result = subprocess.run(
            f"alembic {command}",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        console.print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running alembic {command}:[/red]")
        console.print(e.stderr)
        return e.returncode


@app.command()
def init(
    drop_existing: bool = typer.Option(
        False, "--drop", help="Drop existing tables first"
    )
):
    """Initialize database schema using Alembic migrations."""
    
    if drop_existing:
        console.print("[yellow]Dropping existing tables...[/yellow]")
        run_alembic_command("downgrade base")
    
    console.print("[blue]Running database migrations...[/blue]")
    exit_code = run_alembic_command("upgrade head")
    
    if exit_code == 0:
        console.print("[green]✓ Database initialized successfully![/green]")
    else:
        console.print("[red]✗ Database initialization failed![/red]")
        raise typer.Exit(exit_code)


@app.command()
def migrate():
    """Run pending database migrations."""
    console.print("[blue]Running pending migrations...[/blue]")
    exit_code = run_alembic_command("upgrade head")
    
    if exit_code == 0:
        console.print("[green]✓ Migrations completed successfully![/green]")
    else:
        console.print("[red]✗ Migration failed![/red]")
        raise typer.Exit(exit_code)


@app.command()
def rollback(
    revision: str = typer.Option(
        "-1", help="Revision to rollback to (default: previous)"
    )
):
    """Rollback database migrations."""
    console.print(f"[yellow]Rolling back to revision: {revision}[/yellow]")
    exit_code = run_alembic_command(f"downgrade {revision}")
    
    if exit_code == 0:
        console.print("[green]✓ Rollback completed successfully![/green]")
    else:
        console.print("[red]✗ Rollback failed![/red]")
        raise typer.Exit(exit_code)


@app.command()
def current():
    """Show current database revision."""
    console.print("[blue]Current database revision:[/blue]")
    exit_code = run_alembic_command("current")
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def history():
    """Show migration history."""
    console.print("[blue]Migration history:[/blue]")
    exit_code = run_alembic_command("history")
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def create_migration(
    message: str = typer.Argument(..., help="Migration message"),
    autogenerate: bool = typer.Option(
        True, "--autogenerate/--no-autogenerate", help="Auto-generate migration"
    )
):
    """Create a new migration."""
    command = f'revision {"--autogenerate" if autogenerate else ""} -m "{message}"'
    
    console.print(f"[blue]Creating migration: {message}[/blue]")
    exit_code = run_alembic_command(command)
    
    if exit_code == 0:
        console.print("[green]✓ Migration created successfully![/green]")
    else:
        console.print("[red]✗ Migration creation failed![/red]")
        raise typer.Exit(exit_code)


@app.command()
def test_connection():
    """Test database connection."""
    async def _test():
        try:
            settings = Settings()
            console.print(f"[blue]Testing connection to {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}[/blue]")
            
            # Try to get engine
            engine = get_database_engine(settings)
            console.print("[green]✓ Engine created successfully[/green]")
            
            # Try to create session and execute simple query
            async with get_async_session(settings) as session:
                result = await session.execute(
                    "SELECT version() as version, current_database() as database"
                )
                row = result.fetchone()
                
                console.print(f"[green]✓ Connection successful![/green]")
                console.print(f"Database: {row.database}")
                console.print(f"PostgreSQL version: {row.version}")
                
            await close_database_engine()
            
        except Exception as e:
            console.print(f"[red]✗ Connection failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_test())


@app.command()
def check_tables():
    """Check which tables exist in the database."""
    async def _check():
        try:
            settings = Settings()
            
            async with get_async_session(settings) as session:
                # Query to get all tables
                result = await session.execute("""
                    SELECT table_name, table_type
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = result.fetchall()
                
                if not tables:
                    console.print("[yellow]No tables found in database[/yellow]")
                    return
                
                table = Table()
                table.add_column("Table Name", style="cyan")
                table.add_column("Type", style="magenta")
                
                for row in tables:
                    table.add_row(row.table_name, row.table_type)
                
                console.print(table)
                
                # Check for Alembic version table
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
                    console.print(f"\n[green]Alembic version: {version}[/green]")
                else:
                    console.print("\n[yellow]Alembic not initialized[/yellow]")
            
            await close_database_engine()
            
        except Exception as e:
            console.print(f"[red]✗ Error checking tables: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_check())


@app.command()
def cleanup(
    months: int = typer.Option(6, help="Keep data newer than this many months"),
    dry_run: bool = typer.Option(True, help="Show what would be deleted without deleting")
):
    """Cleanup old sensor readings."""
    async def _cleanup():
        try:
            settings = Settings()
            
            async with get_async_session(settings) as session:
                if dry_run:
                    # Show what would be deleted
                    result = await session.execute(f"""
                        SELECT COUNT(*) as count
                        FROM sensor_readings 
                        WHERE timestamp < NOW() - INTERVAL '{months} months'
                    """)
                    count = result.scalar()
                    console.print(f"[yellow]DRY RUN: Would delete {count} sensor readings older than {months} months[/yellow]")
                else:
                    # Actually delete
                    result = await session.execute(f"""
                        DELETE FROM sensor_readings 
                        WHERE timestamp < NOW() - INTERVAL '{months} months'
                    """)
                    deleted = result.rowcount
                    await session.commit()
                    console.print(f"[green]Deleted {deleted} sensor readings older than {months} months[/green]")
            
            await close_database_engine()
            
        except Exception as e:
            console.print(f"[red]✗ Cleanup failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_cleanup())


@app.command()
def reset():
    """Reset database - drop all tables and run migrations."""
    confirm = typer.confirm("This will destroy all data. Are you sure?")
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    console.print("[red]Resetting database...[/red]")
    
    # Drop everything
    exit_code = run_alembic_command("downgrade base")
    if exit_code != 0:
        console.print("[red]Failed to downgrade[/red]")
        raise typer.Exit(exit_code)
    
    # Recreate
    exit_code = run_alembic_command("upgrade head")
    if exit_code == 0:
        console.print("[green]✓ Database reset successfully![/green]")
    else:
        console.print("[red]✗ Database reset failed![/red]")
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()