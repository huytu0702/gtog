#!/usr/bin/env python3
"""Bootstrap script for Cosmos DB emulator local development."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            [command, "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_tools() -> list[str]:
    """Validate required tools are available."""
    missing = []
    
    if not check_command_exists("docker"):
        missing.append("docker")
    
    # Check docker compose (either as plugin or standalone)
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("docker compose")
    
    return missing


def validate_env_files() -> list[str]:
    """Validate required env files exist."""
    missing = []
    root_dir = Path(__file__).parent.parent.parent
    
    # Check for .env
    env_file = root_dir / ".env"
    if not env_file.exists():
        missing.append(".env")
    
    # Check for .env.cosmos-emulator
    cosmos_env_file = root_dir / ".env.cosmos-emulator"
    if not cosmos_env_file.exists():
        # Check if example exists to suggest
        example_file = root_dir / ".env.cosmos-emulator.example"
        if example_file.exists():
            print(f"Warning: .env.cosmos-emulator not found.")
            print(f"  You can create it by copying: cp .env.cosmos-emulator.example .env.cosmos-emulator")
        missing.append(".env.cosmos-emulator")
    
    return missing


def start_containers() -> bool:
    """Start the cosmos-emulator, backend, and frontend containers."""
    root_dir = Path(__file__).parent.parent.parent
    compose_file = root_dir / "docker-compose.dev.yaml"
    
    print("Starting containers (cosmos-emulator, backend, frontend)...")
    try:
        subprocess.run(
            [
                "docker", "compose", "-f", str(compose_file),
                "up", "-d", "cosmos-emulator", "backend", "frontend"
            ],
            check=True,
            cwd=root_dir,
        )
        print("  Containers started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error starting containers: {e}")
        return False


def wait_for_backend_health(
    base_url: str = "http://localhost:8000",
    timeout: int = 120,
    interval: int = 5,
) -> bool:
    """Poll backend health endpoint until healthy or timeout."""
    health_url = f"{base_url}/health"
    start_time = time.time()
    
    print(f"Waiting for backend to be healthy (timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print(f"  Backend is healthy!")
                    return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            print(f"  Unexpected error: {e}")
        
        time.sleep(interval)
        elapsed = int(time.time() - start_time)
        print(f"  Still waiting... ({elapsed}s elapsed)")
    
    print(f"  Timeout: Backend did not become healthy within {timeout} seconds")
    return False


def print_next_steps():
    """Print next step instructions."""
    print("\n" + "=" * 60)
    print("Bootstrap complete! Next steps:")
    print("=" * 60)
    print()
    print("1. Verify services are running:")
    print("   - Backend API: http://localhost:8000")
    print("   - Frontend:    http://localhost:3000")
    print("   - Cosmos DB:   https://localhost:8081/explorer.html")
    print()
    print("2. Run smoke test:")
    print("   python scripts/cosmos/smoke_api.py")
    print()
    print("3. Run integration tests:")
    print("   pytest backend/tests/integration/test_collections_documents_cosmos.py -v")
    print()
    print("4. Reset everything (removes data):")
    print("   python scripts/cosmos/reset.py")
    print()
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Cosmos DB emulator local development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Full bootstrap with health check
  %(prog)s --skip-health      # Start containers without health check
  %(prog)s --health-timeout 180  # Use 3 minute health check timeout
        """
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check polling (just start containers)",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=120,
        help="Health check timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Base URL for backend API (default: http://localhost:8000)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cosmos DB Emulator Bootstrap")
    print("=" * 60)
    print()
    
    # Step 1: Validate tools
    print("Step 1: Validating required tools...")
    missing_tools = validate_tools()
    if missing_tools:
        print(f"  Error: Missing required tools: {', '.join(missing_tools)}")
        print("  Please install Docker and ensure 'docker compose' is available.")
        sys.exit(1)
    print("  All required tools found")
    print()
    
    # Step 2: Validate env files
    print("Step 2: Validating environment files...")
    missing_env = validate_env_files()
    if missing_env:
        print(f"  Error: Missing required environment files: {', '.join(missing_env)}")
        sys.exit(1)
    print("  All required environment files found")
    print()
    
    # Step 3: Start containers
    print("Step 3: Starting containers...")
    if not start_containers():
        sys.exit(1)
    print()
    
    # Step 4: Health check (unless skipped)
    if not args.skip_health:
        print("Step 4: Health check...")
        if not wait_for_backend_health(
            base_url=args.api_base_url,
            timeout=args.health_timeout,
        ):
            print()
            print("Warning: Health check failed, but containers may still be starting.")
            print("You can check status manually with: docker compose -f docker-compose.dev.yaml ps")
            sys.exit(1)
        print()
    else:
        print("Step 4: Skipping health check (--skip-health)")
        print()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
