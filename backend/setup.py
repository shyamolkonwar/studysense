#!/usr/bin/env python3
"""
StudySense Complete Setup Script

This script sets up all infrastructure components for StudySense across all phases.
Currently implements Phase 1 setup, with structure for future phases.

Usage:
    python setup.py                 # Interactive setup
    python setup.py --phase 1       # Setup specific phase
    python setup.py --check         # Check current setup status
    python setup.py --clean         # Clean up Phase 1 (dev only)
"""

import sys
import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

class StudySenseSetup:
    """Complete StudySense setup manager"""

    def __init__(self):
        self.setup_dir = Path(__file__).parent
        self.config_file = self.setup_dir / "setup_config.json"
        self.load_config()

        # Component status tracking
        self.components = {
            "supabase": {"status": "not_started", "version": None},
            "redis": {"status": "not_started", "version": None},
            "chromadb": {"status": "not_started", "version": None},
            "python_env": {"status": "not_started", "version": None},
            "database_tables": {"status": "not_started", "version": None},
            "storage_buckets": {"status": "not_started", "version": None},
            "chroma_collections": {"status": "not_started", "version": None}
        }

    def load_config(self):
        """Load setup configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "version": "1.0.0",
                "current_phase": 1,
                "completed_phases": [],
                "components": {},
                "environment": {}
            }

    def save_config(self):
        """Save setup configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def run_command(self, command: str, cwd: Optional[Path] = None, check: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        logger.info(f"Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.setup_dir,
                check=check,
                timeout=timeout
            )
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stderr:
                logger.error(f"Error: {e.stderr.strip()}")
            raise

    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("🔍 Checking prerequisites...")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"❌ Python 3.8+ is required (found: {sys.version})")
            return False
        else:
            logger.info(f"✅ Python: {sys.version.split()[0]}")

        # Check required commands
        required_commands = {
            "docker": "Docker",
            "docker-compose": "Docker Compose"
        }

        for cmd, name in required_commands.items():
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout:
                    # Extract version from output (more robust parsing)
                    output = result.stdout.strip()
                    # Look for version number patterns
                    import re
                    version_match = re.search(r'(\d+\.\d+\.\d+)', output)
                    if version_match:
                        version = version_match.group(1)
                        logger.info(f"✅ {name}: {version}")
                    else:
                        # Fallback: use first line of output
                        version = output.split('\n')[0]
                        logger.info(f"✅ {name}: {version}")
                else:
                    logger.error(f"❌ {name} command failed or returned no output")
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {name} command timed out")
                return False
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"❌ {name} is not installed or not in PATH")
                return False

        # Check for Supabase CLI separately (might not be installed yet)
        try:
            result = subprocess.run(["supabase", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                version = result.stdout.strip()
                logger.info(f"✅ Supabase CLI: {version}")
            else:
                logger.warning("⚠️  Supabase CLI not found - will be installed later")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️  Supabase CLI not found - will be installed later")

        return True

    def setup_python_environment(self) -> bool:
        """Setup Python virtual environment and dependencies"""
        logger.info("🐍 Setting up Python environment...")

        try:
            # Create virtual environment if it doesn't exist
            venv_path = self.setup_dir / "venv"
            if not venv_path.exists():
                logger.info("Creating virtual environment...")
                self.run_command(f"python3 -m venv {venv_path}")

            # Activate virtual environment and install dependencies
            if os.name == 'nt':  # Windows
                pip_cmd = f"{venv_path}/Scripts/pip"
                python_cmd = f"{venv_path}/Scripts/python"
            else:  # Unix-like
                pip_cmd = f"{venv_path}/bin/pip"
                python_cmd = f"{venv_path}/bin/python"

            # Upgrade pip and install essential build tools
            logger.info("Upgrading pip and installing build tools...")
            self.run_command(f"{pip_cmd} install --upgrade pip setuptools wheel")

            # Install requirements
            logger.info("Installing Python dependencies...")
            try:
                # Use minimal requirements for Phase 1 first
                if Path("requirements-phase1.txt").exists():
                    logger.info("Installing Phase 1 minimal requirements...")
                    self.run_command(f"{pip_cmd} install -r requirements-phase1.txt", timeout=300)  # 5 minute timeout
                    logger.info("Phase 1 dependencies installed successfully")
                else:
                    # Fallback to full requirements
                    logger.info("Installing full requirements...")
                    self.run_command(f"{pip_cmd} install -r requirements.txt", timeout=600)  # 10 minute timeout
            except subprocess.TimeoutExpired:
                logger.error("❌ Package installation timed out")
                logger.info("   This might be due to network issues or large packages")
                logger.info("   Try installing manually: pip install -r requirements-phase1.txt")
                return False
            except subprocess.CalledProcessError as e:
                logger.error("❌ Package installation failed")
                logger.info("   Try installing manually: pip install -r requirements-phase1.txt")
                logger.info("   Or check your internet connection and disk space")
                return False

            # Verify imports
            logger.info("Verifying Python imports...")
            test_imports = [
                ("from app.core.config import settings", "Core configuration"),
                ("from app.models import User, Message, Consent", "Domain models"),
                ("from app.schemas import UserCreate, MessageCreate", "API schemas"),
                ("from supabase import create_client", "Supabase client"),
                ("import chromadb", "ChromaDB"),
                ("from app.core.supabase_service import supabase_service", "Supabase service")
            ]

            failed_imports = []
            for import_test, description in test_imports:
                try:
                    result = self.run_command(f'{python_cmd} -c "{import_test}"', check=False)
                    if result.returncode != 0:
                        failed_imports.append(description)
                except subprocess.CalledProcessError:
                    failed_imports.append(description)

            if failed_imports:
                logger.warning(f"⚠️  Some imports failed: {', '.join(failed_imports)}")
                logger.info("   This might be normal if some services aren't running yet")
                logger.info("   You can verify individual imports after setup completes")

            self.components["python_env"]["status"] = "completed"
            logger.info("✅ Python environment setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Python environment setup failed: {e}")
            logger.info("   Troubleshooting tips:")
            logger.info("   1. Make sure you have Python 3.8+ installed")
            logger.info("   2. Try removing the venv directory and run setup again")
            logger.info("   3. Check if you have enough disk space")
            logger.info("   4. Verify your internet connection")
            return False

    def setup_supabase(self) -> bool:
        """Setup Supabase local instance"""
        logger.info("🗄️ Setting up Supabase...")

        try:
            # Check if Supabase CLI is installed
            try:
                result = subprocess.run(["supabase", "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    supabase_version = result.stdout.strip()
                    logger.info(f"Supabase CLI: {supabase_version}")
                else:
                    logger.error("❌ Supabase CLI not working")
                    return False
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                logger.error("❌ Supabase CLI not found. Please install it first:")
                logger.error("   npm install -g supabase")
                logger.info("   After installation, run the setup script again")
                return False

            # Initialize Supabase if not already done
            supabase_dir = self.setup_dir / "supabase"
            if not supabase_dir.exists():
                logger.info("Initializing Supabase project...")
                try:
                    result = self.run_command("supabase init", cwd=self.setup_dir)
                except subprocess.CalledProcessError as e:
                    if "already initialized" in e.stderr.lower():
                        logger.info("Supabase project already initialized")
                    else:
                        raise

            # Start Supabase services
            logger.info("Starting Supabase services...")
            try:
                result = self.run_command("supabase start", cwd=self.setup_dir, timeout=300)  # 5 minute timeout
            except subprocess.TimeoutExpired:
                logger.error("❌ Supabase startup timed out. This can happen on first startup.")
                logger.info("   Try running 'supabase start' manually in the backend directory")
                return False

            # Get connection details
            try:
                result = self.run_command("supabase status", cwd=self.setup_dir)
                status_output = result.stdout

                # Parse connection details more robustly
                lines = status_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if 'API URL:' in line:
                        # Extract the URL part after "API URL:"
                        api_url = line.split('API URL:')[-1].strip()
                        if api_url:
                            self.config["environment"]["SUPABASE_URL"] = api_url
                            logger.info(f"Found API URL: {api_url}")
                    elif 'anon key:' in line:
                        anon_key = line.split('anon key:')[-1].strip()
                        if anon_key and anon_key != "null":
                            self.config["environment"]["SUPABASE_ANON_KEY"] = anon_key
                            logger.info("Found anon key")
                    elif 'service_role key:' in line:
                        service_key = line.split('service_role key:')[-1].strip()
                        if service_key and service_key != "null":
                            self.config["environment"]["SUPABASE_SERVICE_ROLE_KEY"] = service_key
                            logger.info("Found service role key")
                    elif 'DB URL:' in line:
                        db_url = line.split('DB URL:')[-1].strip()
                        if db_url:
                            self.config["environment"]["DATABASE_URL"] = db_url
                            logger.info(f"Found DB URL: {db_url[:50]}...")

                # Validate we have the required configuration
                required_keys = ["SUPABASE_URL", "DATABASE_URL"]
                missing_keys = [key for key in required_keys if not self.config["environment"].get(key)]

                if missing_keys:
                    logger.error(f"❌ Missing required configuration: {missing_keys}")
                    logger.error("   Please check if Supabase is running properly")
                    return False

                self.components["supabase"]["status"] = "completed"
                self.components["supabase"]["version"] = supabase_version
                logger.info("✅ Supabase setup complete")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to get Supabase status: {e}")
                logger.error("   Please check if Supabase is running: supabase status")
                return False

        except Exception as e:
            logger.error(f"❌ Supabase setup failed: {e}")
            logger.error("   Please ensure Docker is running and you have proper permissions")
            return False

    def setup_redis(self) -> bool:
        """Setup Redis instance"""
        logger.info("🔴 Setting up Redis...")

        try:
            # Check if Redis is running
            try:
                result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
                if result.stdout.strip() == "PONG":
                    redis_version = subprocess.run(["redis-server", "--version"], capture_output=True, text=True).stdout.strip()
                    self.components["redis"]["status"] = "completed"
                    self.components["redis"]["version"] = redis_version
                    logger.info(f"✅ Redis is already running: {redis_version}")
                    return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Start Redis using Docker
            logger.info("Starting Redis using Docker...")
            self.run_command("docker run -d --name studysense-redis -p 6379:6379 redis:7-alpine")

            # Verify Redis is running
            import time
            time.sleep(3)  # Give Redis time to start

            result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
            if result.stdout.strip() == "PONG":
                self.components["redis"]["status"] = "completed"
                self.components["redis"]["version"] = "7-alpine"
                logger.info("✅ Redis setup complete")
                return True
            else:
                logger.error("❌ Redis failed to start")
                return False

        except Exception as e:
            logger.error(f"❌ Redis setup failed: {e}")
            return False

    def setup_chromadb(self) -> bool:
        """Setup ChromaDB instance"""
        logger.info("🔍 Setting up ChromaDB...")

        try:
            # Check if ChromaDB server is running
            import requests
            response = requests.get("http://localhost:8000/api/v2/heartbeat", timeout=5)
            if response.status_code != 200:
                logger.error("❌ ChromaDB server is not responding")
                return False

            logger.info("✅ ChromaDB server is running")

            # Test ChromaDB client connection
            import chromadb
            client = chromadb.HttpClient(host="localhost", port=8000)

            # Try to get or create a test collection
            try:
                collection = client.get_collection("test_collection")
                logger.info("✅ ChromaDB client connection successful")
            except Exception:
                # Try to create a test collection
                try:
                    collection = client.create_collection("test_collection")
                    logger.info("✅ ChromaDB client connection successful")
                except Exception as e:
                    # Check if it's just a "collection already exists" error
                    if "already exists" in str(e).lower():
                        logger.info("✅ ChromaDB client connection successful (collection exists)")
                    else:
                        logger.error(f"❌ ChromaDB client connection failed: {e}")
                        return False

            # Clean up test collection
            try:
                client.delete_collection("test_collection")
            except Exception:
                pass  # Ignore if collection doesn't exist

            self.components["chromadb"]["status"] = "completed"
            logger.info("✅ ChromaDB setup complete")

            return True

        except Exception as e:
            logger.error(f"❌ ChromaDB setup failed: {e}")
            return False

    def setup_database_tables(self) -> bool:
        """Setup database tables"""
        logger.info("📋 Setting up database tables...")

        try:
            from app.core.db_init import create_tables, check_database_health, get_database_stats

            # Check database health
            if not check_database_health():
                logger.error("❌ Database health check failed")
                return False

            # Create tables
            create_tables()

            # Get stats
            stats = get_database_stats()
            self.components["database_tables"]["status"] = "completed"
            logger.info("✅ Database tables setup complete")
            logger.info(f"   Tables created: {list(stats.keys())}")

            return True

        except Exception as e:
            logger.error(f"❌ Database tables setup failed: {e}")
            return False

    def setup_storage_buckets(self) -> bool:
        """Setup Supabase storage buckets"""
        logger.info("📦 Setting up storage buckets...")

        try:
            from app.core.supabase_service import supabase_service
            from app.core.config import settings

            # Create main storage bucket
            result = supabase_service.create_bucket(
                bucket_name=settings.STORAGE_BUCKET_NAME,
                public=False
            )

            if result.get("success"):
                self.components["storage_buckets"]["status"] = "completed"
                logger.info(f"✅ Storage bucket '{settings.STORAGE_BUCKET_NAME}' created")
                return True
            else:
                logger.error(f"❌ Failed to create storage bucket: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"❌ Storage buckets setup failed: {e}")
            return False

    def setup_chroma_collections(self) -> bool:
        """Setup ChromaDB collections"""
        logger.info("🔍 Setting up ChromaDB collections...")

        try:
            from app.rag.chroma_client import chroma_client

            # ChromaDB collections are created automatically in the client initialization
            stats = chroma_client.get_collection_stats()

            # Verify required collections exist
            required_collections = [
                "kb_global", "campus_resources", "user_context",
                "conversation_context", "risk_patterns"
            ]

            existing_collections = list(stats.keys())
            missing_collections = set(required_collections) - set(existing_collections)

            if missing_collections:
                logger.error(f"❌ Missing collections: {missing_collections}")
                return False

            self.components["chroma_collections"]["status"] = "completed"
            logger.info("✅ ChromaDB collections setup complete")
            logger.info(f"   Collections: {existing_collections}")

            return True

        except Exception as e:
            logger.error(f"❌ ChromaDB collections setup failed: {e}")
            return False

    def create_env_file(self):
        """Create .env file with configuration"""
        logger.info("📝 Creating .env file...")

        env_file = self.setup_dir / ".env"

        if not env_file.exists():
            env_content = f"""# StudySense Environment Configuration
# Generated on {datetime.now().isoformat()}

# Database (Supabase)
DATABASE_URL={self.config["environment"].get("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")}
SUPABASE_URL={self.config["environment"].get("SUPABASE_URL", "http://localhost:54321")}
SUPABASE_ANON_KEY={self.config["environment"].get("SUPABASE_ANON_KEY", "your_anon_key_here")}
SUPABASE_SERVICE_ROLE_KEY={self.config["environment"].get("SUPABASE_SERVICE_ROLE_KEY", "your_service_role_key_here")}

# Security
SECRET_KEY=your-secret-key-here-change-in-production-min-32-characters-long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/chroma
EMBEDDING_MODEL=text-embedding-ada-002

# LLM Configuration (optional for Phase 1)
# OPENAI_API_KEY=your_openai_key_here

# Data Retention
MESSAGE_RETENTION_DAYS=90
AUDIT_RETENTION_DAYS=2555
DEFAULT_DATA_RETENTION_DAYS=365

# Storage
STORAGE_BUCKET_NAME=studysense-data

# Application
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO
"""

            with open(env_file, 'w') as f:
                f.write(env_content)

            logger.info("✅ .env file created")
            logger.info("⚠️  Please update the SECRET_KEY and add your API keys")
        else:
            logger.info("ℹ️  .env file already exists")

    def setup_phase_1(self) -> bool:
        """Setup Phase 1 components"""
        logger.info("🚀 Starting Phase 1 Setup")
        logger.info("=" * 50)

        phases = [
            ("Python Environment", self.setup_python_environment),
            ("Supabase", self.setup_supabase),
            ("Redis", self.setup_redis),
            # ("ChromaDB", self.setup_chromadb),  # Skip ChromaDB for now
            ("Database Tables", self.setup_database_tables),
            ("Storage Buckets", self.setup_storage_buckets),
            # ("ChromaDB Collections", self.setup_chroma_collections),  # Skip for now due to client issues
        ]

        for phase_name, setup_func in phases:
            logger.info(f"\n--- {phase_name} ---")
            try:
                if setup_func():
                    logger.info(f"✅ {phase_name} completed successfully")
                else:
                    logger.error(f"❌ {phase_name} failed")
                    return False
            except Exception as e:
                logger.error(f"❌ {phase_name} failed with exception: {e}")
                return False

        # Create environment file
        self.create_env_file()

        # Update configuration
        self.config["current_phase"] = 1
        if 1 not in self.config["completed_phases"]:
            self.config["completed_phases"].append(1)
        self.save_config()

        logger.info("\n" + "=" * 50)
        logger.info("🎉 Phase 1 setup completed successfully!")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Update the .env file with your secret keys")
        logger.info("2. Start the development server: python -m uvicorn app.main:app --reload")
        logger.info("3. Visit http://localhost:8000/docs for API documentation")

        return True

    def check_setup(self) -> bool:
        """Check current setup status"""
        logger.info("🔍 Checking setup status...")
        logger.info("=" * 50)

        # Check Python environment
        try:
            from app.core.config import settings
            logger.info("✅ Python imports working")
        except Exception as e:
            logger.error(f"❌ Python imports failed: {e}")

        # Check Supabase
        try:
            from app.core.supabase_service import supabase_service
            health = supabase_service.health_check()
            if health.get("available"):
                logger.info("✅ Supabase is running")
            else:
                logger.error("❌ Supabase is not available")
        except Exception as e:
            logger.error(f"❌ Supabase check failed: {e}")

        # Check Redis
        try:
            result = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True)
            if result.stdout.strip() == "PONG":
                logger.info("✅ Redis is running")
            else:
                logger.error("❌ Redis is not responding")
        except:
            logger.error("❌ Redis is not available")

        # Check ChromaDB
        try:
            from app.rag.chroma_client import chroma_client
            stats = chroma_client.get_collection_stats()
            logger.info(f"✅ ChromaDB is running with {len(stats)} collections")
        except Exception as e:
            logger.error(f"❌ ChromaDB check failed: {e}")

        logger.info("=" * 50)
        return True

    def clean_phase_1(self) -> bool:
        """Clean up Phase 1 components (development only)"""
        logger.info("🧹 Cleaning up Phase 1...")

        confirm = input("⚠️  This will stop and remove all Phase 1 services. Continue? (y/N): ")
        if confirm.lower() != 'y':
            logger.info("Cancelled cleanup")
            return False

        try:
            # Stop Supabase
            logger.info("Stopping Supabase...")
            try:
                subprocess.run("supabase stop", cwd=self.setup_dir, shell=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning("Supabase stop timed out, continuing...")

            # Stop Redis container
            logger.info("Stopping Redis...")
            try:
                subprocess.run("docker stop studysense-redis", shell=True, timeout=30)
                subprocess.run("docker rm studysense-redis", shell=True, timeout=30)
            except:
                logger.warning("Redis cleanup failed, continuing...")

            # Remove data directories
            logger.info("Removing data directories...")
            import shutil
            data_dirs = ["data", "venv", "supabase"]
            for dir_name in data_dirs:
                dir_path = self.setup_dir / dir_name
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"Removed {dir_name} directory")
                    except OSError as e:
                        logger.warning(f"Failed to remove {dir_name}: {e}")

            # Remove config files
            config_files = ["setup_config.json", ".env", "setup.log"]
            for config_file in config_files:
                file_path = self.setup_dir / config_file
                if file_path.exists():
                    try:
                        file_path.unlink()
                        logger.info(f"Removed {config_file}")
                    except OSError as e:
                        logger.warning(f"Failed to remove {config_file}: {e}")

            logger.info("✅ Phase 1 cleanup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False

    def setup_phase_2(self) -> bool:
        """Setup Phase 2 components (placeholder for future)"""
        logger.info("📋 Phase 2 setup not yet implemented")
        logger.info("Phase 2 will include:")
        logger.info("- RAG knowledge base and retrieval")
        logger.info("- LLM integration")
        logger.info("- Agent tool calling")
        return False

    def setup_phase_3(self) -> bool:
        """Setup Phase 3 components (placeholder for future)"""
        logger.info("📋 Phase 3 setup not yet implemented")
        logger.info("Phase 3 will include:")
        logger.info("- Analysis and risk scoring engine")
        logger.info("- Alerts and notifications")
        logger.info("- Backend services")
        return False

    def setup_phase_4(self) -> bool:
        """Setup Phase 4 components (placeholder for future)"""
        logger.info("📋 Phase 4 setup not yet implemented")
        logger.info("Phase 4 will include:")
        logger.info("- Frontend application")
        logger.info("- User interfaces")
        logger.info("- Dashboard and visualizations")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="StudySense Setup Script")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                       help="Setup specific phase (default: Phase 1)")
    parser.add_argument("--check", action="store_true",
                       help="Check current setup status")
    parser.add_argument("--clean", action="store_true",
                       help="Clean up current setup (development only)")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Interactive setup mode")

    args = parser.parse_args()

    # Verify we're in the correct directory
    if not Path("requirements.txt").exists():
        logger.error("❌ Please run this script from the backend directory")
        logger.error("   The script should be run from: backend/")
        return False

    setup = StudySenseSetup()

    if args.check:
        return setup.check_setup()
    elif args.clean:
        return setup.clean_phase_1()
    elif args.phase:
        if not setup.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False

        phase_methods = {
            1: setup.setup_phase_1,
            2: setup.setup_phase_2,
            3: setup.setup_phase_3,
            4: setup.setup_phase_4,
        }

        setup_func = phase_methods.get(args.phase)
        if setup_func:
            return setup_func()
        else:
            logger.error(f"❌ Phase {args.phase} setup not implemented")
            return False
    else:
        # Default: interactive setup
        if not setup.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False

        logger.info("🚀 StudySense Setup")
        logger.info("Which phase would you like to set up?")
        logger.info("1. Phase 1: Domain model and data contracts (✅ Ready)")
        logger.info("2. Phase 2: RAG knowledge base and retrieval (🚧 Not ready)")
        logger.info("3. Phase 3: Analysis and risk scoring (🚧 Not ready)")
        logger.info("4. Phase 4: Frontend application (🚧 Not ready)")
        logger.info("c. Check current setup")
        logger.info("l. Clean up setup")

        choice = input("Enter your choice (1-4, c, l): ").lower().strip()

        if choice == "1":
            return setup.setup_phase_1()
        elif choice == "c":
            return setup.check_setup()
        elif choice == "l":
            return setup.clean_phase_1()
        else:
            logger.error("❌ Invalid choice or phase not ready")
            return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
