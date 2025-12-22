//! Face Pattern Persistence
//!
//! Stores face signatures for visual identity verification.
//! Integrates with sam-audio's face recognition for multi-modal identity.

use frame_catalog::database::Database;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type Result<T> = std::result::Result<T, FaceStoreError>;

#[derive(Debug, thiserror::Error)]
pub enum FaceStoreError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("DateTime parse error: {0}")]
    DateTimeParse(#[from] chrono::ParseError),

    #[error("Face signature not found for user: {0}")]
    NotFound(String),

    #[error("Cannot modify immutable face signature for user: {0}")]
    ImmutableSignature(String),
}

/// Face signature stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceSignature {
    pub user_id: String,
    pub mean_embedding: Vec<f32>, // 512-dim
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub immutable: bool,
}

/// Face pattern storage and retrieval
pub struct FaceStore {
    db: Arc<Mutex<Connection>>,
}

impl FaceStore {
    /// Create new face store
    pub fn new(database: &Database) -> Result<Self> {
        let db = database.conn();

        // Initialize schema
        {
            let conn = db.lock().unwrap();

            // Face signatures table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS face_signatures (
                    user_id TEXT PRIMARY KEY,
                    mean_embedding TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    last_updated TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    tags TEXT,
                    immutable INTEGER NOT NULL DEFAULT 0
                )",
                [],
            )?;

            // Immutability protection triggers
            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS prevent_face_immutable_update
                 BEFORE UPDATE ON face_signatures
                 FOR EACH ROW
                 WHEN OLD.immutable = 1
                 BEGIN
                     SELECT RAISE(ABORT, 'Cannot modify immutable face signature');
                 END",
                [],
            )?;

            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS prevent_face_immutable_delete
                 BEFORE DELETE ON face_signatures
                 FOR EACH ROW
                 WHEN OLD.immutable = 1
                 BEGIN
                     SELECT RAISE(ABORT, 'Cannot delete immutable face signature');
                 END",
                [],
            )?;
        }

        Ok(Self { db })
    }

    /// Store face signature
    pub fn store_signature(&self, signature: &FaceSignature) -> Result<()> {
        let conn = self.db.lock().unwrap();

        // Check if signature exists and is immutable
        let existing: Option<i32> = conn
            .query_row(
                "SELECT immutable FROM face_signatures WHERE user_id = ?1",
                params![&signature.user_id],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(is_immutable) = existing {
            if is_immutable == 1 {
                return Err(FaceStoreError::ImmutableSignature(
                    signature.user_id.clone(),
                ));
            }
        }

        let embedding_json = serde_json::to_string(&signature.mean_embedding)?;
        let metadata_json = signature
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m))
            .transpose()?;
        let tags_json = signature
            .tags
            .as_ref()
            .map(|t| serde_json::to_string(t))
            .transpose()?;

        conn.execute(
            "INSERT OR REPLACE INTO face_signatures (
                user_id, mean_embedding, sample_count, last_updated,
                metadata, tags, immutable
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                signature.user_id,
                embedding_json,
                signature.sample_count as i64,
                signature.last_updated.to_rfc3339(),
                metadata_json,
                tags_json,
                if signature.immutable { 1 } else { 0 },
            ],
        )?;

        Ok(())
    }

    /// Get face signature for user
    pub fn get_signature(&self, user_id: &str) -> Result<Option<FaceSignature>> {
        let conn = self.db.lock().unwrap();

        conn.query_row(
            "SELECT user_id, mean_embedding, sample_count, last_updated,
                    metadata, tags, immutable
             FROM face_signatures WHERE user_id = ?1",
            params![user_id],
            |row| Self::signature_from_row(row),
        )
        .optional()
        .map_err(FaceStoreError::from)
    }

    /// Get all face signatures
    pub fn get_all_signatures(&self) -> Result<HashMap<String, FaceSignature>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT user_id, mean_embedding, sample_count, last_updated,
                    metadata, tags, immutable
             FROM face_signatures",
        )?;

        let signatures = stmt
            .query_map([], |row| Self::signature_from_row(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(signatures
            .into_iter()
            .map(|sig| (sig.user_id.clone(), sig))
            .collect())
    }

    /// Count face signatures
    pub fn count(&self) -> Result<usize> {
        let conn = self.db.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM face_signatures", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    // Helper: Parse signature from row
    fn signature_from_row(row: &Row) -> rusqlite::Result<FaceSignature> {
        let embedding_json: String = row.get(1)?;
        let last_updated_str: String = row.get(3)?;
        let metadata_json: Option<String> = row.get(4)?;
        let tags_json: Option<String> = row.get(5)?;
        let immutable: i32 = row.get(6)?;

        let mean_embedding: Vec<f32> = serde_json::from_str(&embedding_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(1, rusqlite::types::Type::Text, Box::new(e))
        })?;

        let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?
            .with_timezone(&Utc);

        let metadata = metadata_json
            .as_ref()
            .map(|json| serde_json::from_str(json))
            .transpose()
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;

        let tags = tags_json
            .as_ref()
            .map(|json| serde_json::from_str(json))
            .transpose()
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;

        Ok(FaceSignature {
            user_id: row.get(0)?,
            mean_embedding,
            sample_count: row.get::<_, i64>(2)? as usize,
            last_updated,
            metadata,
            tags,
            immutable: immutable != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame_catalog::database::Database;

    fn create_test_signature(user_id: &str) -> FaceSignature {
        FaceSignature {
            user_id: user_id.to_string(),
            mean_embedding: vec![0.1; 512],
            sample_count: 1,
            last_updated: Utc::now(),
            metadata: None,
            tags: None,
            immutable: false,
        }
    }

    #[test]
    fn test_face_store_creation() {
        let db = Database::new(":memory:").unwrap();
        let store = FaceStore::new(&db).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let db = Database::new(":memory:").unwrap();
        let store = FaceStore::new(&db).unwrap();

        let signature = create_test_signature("test_user");
        store.store_signature(&signature).unwrap();

        let retrieved = store.get_signature("test_user").unwrap();
        assert!(retrieved.is_some());

        let retrieved_sig = retrieved.unwrap();
        assert_eq!(retrieved_sig.user_id, "test_user");
        assert_eq!(retrieved_sig.mean_embedding.len(), 512);
        assert_eq!(retrieved_sig.sample_count, 1);
    }

    #[test]
    fn test_immutable_protection() {
        let db = Database::new(":memory:").unwrap();
        let store = FaceStore::new(&db).unwrap();

        let mut signature = create_test_signature("immutable_user");
        signature.immutable = true;

        store.store_signature(&signature).unwrap();

        // Try to update (should fail)
        let mut modified = signature.clone();
        modified.sample_count = 999;

        let result = store.store_signature(&modified);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("immutable"));
    }

    #[test]
    fn test_get_all_signatures() {
        let db = Database::new(":memory:").unwrap();
        let store = FaceStore::new(&db).unwrap();

        store
            .store_signature(&create_test_signature("user1"))
            .unwrap();
        store
            .store_signature(&create_test_signature("user2"))
            .unwrap();

        let all = store.get_all_signatures().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("user1"));
        assert!(all.contains_key("user2"));
    }
}
