from celery import Celery
from kombu import Queue
import os
from app.core.config import settings

# Configure Celery
celery_app = Celery(
    'studysense',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.analysis_tasks',
        'app.tasks.notification_tasks',
        'app.tasks.data_ingestion_tasks'
    ]
)

# Configure task routing and queues
celery_app.conf.update(
    # Task routing configuration
    task_routes={
        'app.tasks.analysis_tasks.analyze_user_stress': {'queue': 'analysis'},
        'app.tasks.analysis_tasks.process_daily_batch': {'queue': 'batch'},
        'app.tasks.notification_tasks.send_alert_notifications': {'queue': 'notifications'},
        'app.tasks.notification_tasks.process_escalation': {'queue': 'escalation'},
        'app.tasks.data_ingestion_tasks.ingest_user_messages': {'queue': 'ingestion'},
        'app.tasks.data_ingestion_tasks.process_calendar_events': {'queue': 'ingestion'},
        'app.tasks.analysis_tasks.update_knowledge_base': {'queue': 'maintenance'}
    },

    # Queue definitions
    task_queues=(
        Queue('analysis', routing_key='analysis'),
        Queue('batch', routing_key='batch'),
        Queue('notifications', routing_key='notifications'),
        Queue('escalation', routing_key='escalation'),
        Queue('ingestion', routing_key='ingestion'),
        Queue('maintenance', routing_key='maintenance'),
    ),

    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Timeouts and retries
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    task_acks_late=True,
    worker_prefetch_multiplier=4,

    # Result expiration (1 day)
    result_expires=86400,

    # Beat schedule configuration
    beat_schedule={
        'daily-user-analysis': {
            'task': 'app.tasks.analysis_tasks.process_daily_batch',
            'schedule': 300.0,  # Every 5 minutes
        },
        'knowledge-base-update': {
            'task': 'app.tasks.analysis_tasks.update_knowledge_base',
            'schedule': 3600.0,  # Every hour
        }
    }
)

# Configure monitoring and error handling
celery_app.conf.update(
    # Task monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,

    # Optimization
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_connection_retry_delay=5,

    # Logging
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_log_colorize=False,
)

# Environment-specific configuration
if os.getenv('ENVIRONMENT') == 'production':
    celery_app.conf.update(
        # Production settings
        worker_concurrency=4,
        worker_prefetch_multiplier=2,
        task_compression='gzip',
    )
else:
    # Development settings
    celery_app.conf.update(
        # Development settings
        worker_concurrency=2,
        worker_log_level='INFO',
        task_eager_propagates=True,
    )