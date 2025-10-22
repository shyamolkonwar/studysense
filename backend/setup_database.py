#!/usr/bin/env python3
"""
Database Setup Script for StudySense
Sets up Supabase database tables and initial data
"""

import sys
import os
import logging
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main database setup function"""
    logger.info("🚀 Starting StudySense Database Setup")
    logger.info("=" * 50)

    try:
        # Import database initialization functions
        from app.core.db_init import create_tables, check_database_health, get_database_stats

        # Check database health first
        logger.info("🔍 Checking database connectivity...")
        if not check_database_health():
            logger.error("❌ Database health check failed!")
            logger.error("   Please ensure Supabase is running and connection details are correct")
            return False

        # Create tables and initial data
        logger.info("📋 Creating database tables...")
        create_tables()

        # Get final stats
        logger.info("📊 Getting database statistics...")
        stats = get_database_stats()

        logger.info("\n" + "=" * 50)
        logger.info("🎉 Database setup completed successfully!")
        logger.info("\n📋 Database Summary:")
        for table, count in stats.items():
            if table != 'database_size':
                logger.info(f"   {table}: {count} records")
        if 'database_size' in stats:
            logger.info(f"   Database size: {stats['database_size']}")

        logger.info("\n📋 Next Steps:")
        logger.info("1. Verify the tables were created in your Supabase dashboard")
        logger.info("2. Check that initial data was inserted correctly")
        logger.info("3. Run the main setup script: python setup.py --phase 1")

        return True

    except Exception as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        logger.error("\n🔧 Troubleshooting:")
        logger.error("1. Make sure Supabase is running: supabase status")
        logger.error("2. Check your .env file has correct database credentials")
        logger.error("3. Verify your internet connection")
        logger.error("4. Try running: python -c 'from app.core.db_init import check_database_health; check_database_health()'")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
