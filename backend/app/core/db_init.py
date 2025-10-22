"""
Database initialization script for StudySense Phase 1
Creates all tables and sets up initial data
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.database import Base, get_db, engine
from app.core.config import settings
from app.models import *  # Import all models to ensure they're registered
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Tables created successfully!")

        # Create indexes for better performance
        create_indexes()

        # Insert initial data
        insert_initial_data()

    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

def create_indexes():
    """Create additional indexes for performance optimization"""
    indexes = [
        # User-related indexes
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
        "CREATE INDEX IF NOT EXISTS idx_users_institution ON users(institution);",
        "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);",
        "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);",

        # Message-related indexes
        "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);",
        "CREATE INDEX IF NOT EXISTS idx_messages_sentiment_score ON messages(sentiment_score);",
        "CREATE INDEX IF NOT EXISTS idx_messages_stress_indicators ON messages(stress_indicators);",
        "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel);",
        "CREATE INDEX IF NOT EXISTS idx_messages_content_hash ON messages(content_hash);",
        "CREATE INDEX IF NOT EXISTS idx_messages_user_timestamp ON messages(user_id, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_messages_processed_at ON messages(processed_at);",

        # Activity-related indexes
        "CREATE INDEX IF NOT EXISTS idx_activities_user_id ON activities(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_activities_start_time ON activities(start_time);",
        "CREATE INDEX IF NOT EXISTS idx_activities_type ON activities(activity_type);",
        "CREATE INDEX IF NOT EXISTS idx_activities_user_type_time ON activities(user_id, activity_type, start_time DESC);",
        "CREATE INDEX IF NOT EXISTS idx_activities_anomaly ON activities(is_anomaly);",

        # Risk score indexes
        "CREATE INDEX IF NOT EXISTS idx_risk_scores_user_id ON risk_scores(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_risk_scores_calculated_at ON risk_scores(calculated_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_risk_scores_level ON risk_scores(risk_level);",
        "CREATE INDEX IF NOT EXISTS idx_risk_scores_user_calculated ON risk_scores(user_id, calculated_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_risk_scores_alert_triggered ON risk_scores(alert_triggered);",

        # Alert-related indexes
        "CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);",
        "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);",
        "CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);",
        "CREATE INDEX IF NOT EXISTS idx_alerts_user_status ON alerts(user_id, status);",

        # Consent-related indexes
        "CREATE INDEX IF NOT EXISTS idx_consents_user_id ON consents(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_consents_status ON consents(status);",
        "CREATE INDEX IF NOT EXISTS idx_consents_type ON consents(consent_type);",
        "CREATE INDEX IF NOT EXISTS idx_consents_expires_at ON consents(expires_at);",

        # Integration-related indexes
        "CREATE INDEX IF NOT EXISTS idx_integrations_user_id ON integration_accounts(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_integrations_provider ON integration_accounts(provider);",
        "CREATE INDEX IF NOT EXISTS idx_integrations_status ON integration_accounts(status);",
        "CREATE INDEX IF NOT EXISTS idx_integrations_last_sync ON integration_accounts(last_sync_at DESC);",

        # Audit log indexes
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);",
    ]

    try:
        with engine.connect() as conn:
            for index_sql in indexes:
                conn.execute(text(index_sql))
            conn.commit()
        logger.info("Additional indexes created successfully!")
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        raise

def insert_initial_data():
    """Insert initial system data"""
    try:
        with engine.connect() as conn:
            # Create additional tables if they don't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS legal_versions (
                    version VARCHAR(20) PRIMARY KEY,
                    effective_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    privacy_policy_url TEXT,
                    terms_url TEXT
                );
            """))

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS notification_templates (
                    id SERIAL PRIMARY KEY,
                    type VARCHAR(100) UNIQUE NOT NULL,
                    template TEXT NOT NULL,
                    default_enabled BOOLEAN DEFAULT true,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """))

            # Insert legal versions for consent
            conn.execute(text("""
                INSERT INTO legal_versions (version, effective_date, privacy_policy_url, terms_url)
                VALUES
                    ('1.0', NOW(), '/privacy-policy/v1.0', '/terms/v1.0')
                ON CONFLICT (version) DO NOTHING;
            """))

            # Insert default notification preferences
            conn.execute(text("""
                INSERT INTO notification_templates (type, template, default_enabled)
                VALUES
                    ('risk_alert', 'Your stress level has changed to {risk_level}', true),
                    ('weekly_report', 'Your weekly wellness report is ready', true),
                    ('deadline_reminder', 'You have {count} upcoming deadlines', true),
                    ('consent_expiry', 'Your consent for {consent_type} expires in {days} days', true)
                ON CONFLICT (type) DO NOTHING;
            """))

            conn.commit()
        logger.info("Initial data inserted successfully!")
    except Exception as e:
        logger.error(f"Error inserting initial data: {str(e)}")
        raise

def reset_database():
    """Drop and recreate all tables (development only)"""
    if settings.ENVIRONMENT != "development":
        raise ValueError("Database reset is only allowed in development environment")

    try:
        logger.warning("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped!")

        create_tables()
        logger.info("Database reset completed!")
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        raise

def check_database_health():
    """Check database connectivity and basic health"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as health_check"))
            row = result.fetchone()

            if row and row[0] == 1:
                logger.info("Database health check passed!")
                return True
            else:
                logger.error("Database health check failed!")
                return False
    except Exception as e:
        logger.error(f"Database health check error: {str(e)}")
        return False

def get_database_stats():
    """Get basic database statistics"""
    try:
        with engine.connect() as conn:
            stats = {}

            # Get table counts
            tables = ['users', 'messages', 'activities', 'risk_scores', 'alerts', 'consents', 'integration_accounts']
            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = result.fetchone()[0]

            # Get database size
            result = conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database())"))
            stats['database_size'] = result.fetchone()[0]

            logger.info(f"Database stats: {stats}")
            return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "create":
            create_tables()
        elif command == "reset":
            reset_database()
        elif command == "health":
            check_database_health()
        elif command == "stats":
            get_database_stats()
        else:
            print("Usage: python db_init.py [create|reset|health|stats]")
    else:
        # Default action: create tables
        create_tables()
