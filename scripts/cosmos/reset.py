#!/usr/bin/env python3
"""Reset script for Cosmos DB emulator environment."""

import argparse
import subprocess
import sys
from pathlib import Path


def reset_containers(clean_local_storage: bool = False) -> bool:
    """Reset Docker containers and volumes."""
    root_dir = Path(__file__).parent.parent.parent
    compose_file = root_dir / "docker-compose.dev.yaml"
    
    print("Stopping and removing containers...")
    try:
        subprocess.run(
            [
                "docker", "compose", "-f", str(compose_file),
                "down", "-v"
            ],
            check=True,
            cwd=root_dir,
        )
        print("  Containers stopped and volumes removed")
    except subprocess.CalledProcessError as e:
        print(f"  Error stopping containers: {e}")
        return False
    
    if clean_local_storage:
        print("Cleaning local storage...")
        storage_dir = root_dir / "backend" / "storage"
        if storage_dir.exists():
            try:
                # Remove all contents but keep the directory
                for item in storage_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                print("  Local storage cleaned")
            except Exception as e:
                print(f"  Warning: Could not clean local storage: {e}")
        else:
            print("  Local storage directory does not exist")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reset Cosmos DB emulator environment (removes all data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This will DELETE all data in the Cosmos DB emulator and remove containers!

Examples:
  %(prog)s --yes                    # Reset with confirmation flag
  %(prog)s --yes --clean-local-storage  # Also clean local backend storage

When to use reset:
  - Corrupted data or inconsistent state
  - Starting fresh after configuration changes
  - Cleaning up after testing
  - Troubleshooting connection issues

After reset, run bootstrap to restart:
  python scripts/cosmos/bootstrap.py
        """
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        required=True,
        help="Confirm destructive action (required)",
    )
    parser.add_argument(
        "--clean-local-storage",
        action="store_true",
        help="Also remove local backend storage files",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COSMOS DB EMULATOR RESET")
    print("=" * 60)
    print()
    print("WARNING: This will:")
    print("  1. Stop all running containers (cosmos-emulator, backend, frontend)")
    print("  2. REMOVE all Docker volumes (including Cosmos DB data)")
    if args.clean_local_storage:
        print("  3. REMOVE local backend storage files")
    print()
    print("All data will be lost!")
    print()
    
    if not args.yes:
        print("Error: --yes flag is required to proceed")
        print()
        print("To confirm reset, run:")
        print("  python scripts/cosmos/reset.py --yes")
        sys.exit(1)
    
    print("Proceeding with reset...")
    print()
    
    if reset_containers(clean_local_storage=args.clean_local_storage):
        print()
        print("=" * 60)
        print("RESET COMPLETE")
        print("=" * 60)
        print()
        print("All data has been removed.")
        print()
        print("To restart the environment:")
        print("  python scripts/cosmos/bootstrap.py")
        print()
    else:
        print()
        print("RESET FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
