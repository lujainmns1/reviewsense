-- Manual SQL migration to add 'name' column to analysis_sessions table
-- Run this script directly on your PostgreSQL database if you prefer manual migration

-- Add name column to analysis_sessions table
ALTER TABLE analysis_sessions 
ADD COLUMN IF NOT EXISTS name VARCHAR(255);

-- Add comment to column
COMMENT ON COLUMN analysis_sessions.name IS 'User-defined name for the analysis session';

-- Verify the column was added
SELECT column_name, data_type, character_maximum_length, is_nullable
FROM information_schema.columns
WHERE table_name = 'analysis_sessions' AND column_name = 'name';

