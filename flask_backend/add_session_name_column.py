"""
Migration script to add 'name' column to analysis_sessions table.

This script can be run in two ways:
1. As a Flask-Migrate migration (recommended)
2. As a standalone SQL script

To run as Flask-Migrate migration:
    flask db migrate -m "Add name column to analysis_sessions"
    flask db upgrade

To run as standalone SQL (if you prefer manual migration):
    Run the SQL commands below directly on your database.
"""

from flask import current_app
from flask_migrate import upgrade, downgrade
from app import app, db

# SQL for manual migration (PostgreSQL)
MANUAL_MIGRATION_SQL = """
-- Add name column to analysis_sessions table
ALTER TABLE analysis_sessions 
ADD COLUMN IF NOT EXISTS name VARCHAR(255);

-- Add comment to column
COMMENT ON COLUMN analysis_sessions.name IS 'User-defined name for the analysis session';
"""

# SQL for rollback (if needed)
ROLLBACK_SQL = """
-- Remove name column from analysis_sessions table
ALTER TABLE analysis_sessions 
DROP COLUMN IF EXISTS name;
"""


def upgrade_database():
    """Run the migration using Flask-Migrate"""
    with app.app_context():
        try:
            from sqlalchemy import text
            
            # Check if column already exists
            with db.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='analysis_sessions' AND column_name='name'
                """))
                if result.fetchone():
                    print("✓ Column 'name' already exists in analysis_sessions table")
                    return
                
                # Add the column
                conn.execute(text("""
                    ALTER TABLE analysis_sessions 
                    ADD COLUMN IF NOT EXISTS name VARCHAR(255)
                """))
                conn.commit()
            
            print("✓ Successfully added 'name' column to analysis_sessions table")
        except Exception as e:
            print(f"✗ Error running migration: {e}")
            db.session.rollback()
            raise


def downgrade_database():
    """Rollback the migration"""
    with app.app_context():
        try:
            from sqlalchemy import text
            
            with db.engine.connect() as conn:
                conn.execute(text("""
                    ALTER TABLE analysis_sessions 
                    DROP COLUMN IF EXISTS name
                """))
                conn.commit()
            
            print("✓ Successfully removed 'name' column from analysis_sessions table")
        except Exception as e:
            print(f"✗ Error rolling back migration: {e}")
            db.session.rollback()
            raise


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'downgrade':
        print("Rolling back migration...")
        downgrade_database()
    else:
        print("Running migration to add 'name' column...")
        upgrade_database()
        print("\nMigration completed!")
        print("\nTo rollback, run: python add_session_name_column.py downgrade")

