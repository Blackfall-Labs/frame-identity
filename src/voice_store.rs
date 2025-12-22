//! Voice Pattern Persistence
//!
//! Stores voice signatures and samples for speaker identification.
//! Integrates with sam-audio's voice recognition for multi-modal identity verification.

use sam_vector::database::Database;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type Result<T> = std::result::Result<T, VoiceStoreError>;

#[derive(Debug, thiserror::Error)]
pub enum VoiceStoreError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("DateTime parse error: {0}")]
    DateTimeParse(#[from] chrono::ParseError),

    #[error("Voice signature not found for user: {0}")]
    NotFound(String),

    #[error("Cannot modify immutable voice signature for user: {0}")]
    ImmutableSignature(String),
}

/// Voice signature stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSignature {
    pub user_id: String,
    pub fundamental_frequency: f32,
    pub frequency_range: (f32, f32),
    pub formant_frequencies: Vec<f32>,
    pub speech_rate: f32,
    pub mfcc_signature: Vec<f32>,
    pub spectral_centroid: f32,
    pub jitter: f32,
    pub shimmer: f32,
    pub harmonic_to_noise_ratio: f32,
    pub sample_count: usize,
    pub last_updated: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub immutable: bool,
}

/// Voice sample for incremental learning
#[derive(Debug, Clone)]
pub struct VoiceSample {
    pub id: String,
    pub user_id: String,
    pub audio_data: Vec<u8>, // Compressed audio (optional for retraining)
    pub duration_ms: u32,
    pub sample_rate: u32,
    pub fundamental_frequency: f32,
    pub formant_frequencies: Vec<f32>,
    pub mfcc: Vec<f32>,
    pub quality_score: f32,
    pub timestamp: DateTime<Utc>,
}

/// Voice pattern storage and retrieval
pub struct VoiceStore {
    db: Arc<Mutex<Connection>>,
}

impl VoiceStore {
    /// Create new voice store
    pub fn new(database: &Database) -> Result<Self> {
        let db = database.conn();

        // Initialize schema
        {
            let conn = db.lock().unwrap();

            // Voice signatures table
            conn.execute(
                "CREATE TABLE IF NOT EXISTS voice_signatures (
                user_id TEXT PRIMARY KEY,
                fundamental_frequency REAL NOT NULL,
                frequency_range_min REAL NOT NULL,
                frequency_range_max REAL NOT NULL,
                formant_frequencies TEXT NOT NULL,
                speech_rate REAL NOT NULL,
                mfcc_signature TEXT NOT NULL,
                spectral_centroid REAL NOT NULL,
                jitter REAL NOT NULL,
                shimmer REAL NOT NULL,
                harmonic_to_noise_ratio REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                tags TEXT,
                immutable INTEGER NOT NULL DEFAULT 0
            )",
                [],
            )?;

            // Immutability protection trigger
            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS prevent_immutable_update
             BEFORE UPDATE ON voice_signatures
             FOR EACH ROW
             WHEN OLD.immutable = 1
             BEGIN
                 SELECT RAISE(ABORT, 'Cannot modify immutable voice signature');
             END",
                [],
            )?;

            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS prevent_immutable_delete
             BEFORE DELETE ON voice_signatures
             FOR EACH ROW
             WHEN OLD.immutable = 1
             BEGIN
                 SELECT RAISE(ABORT, 'Cannot delete immutable voice signature');
             END",
                [],
            )?;

            // Voice samples table (for incremental learning)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS voice_samples (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                audio_data BLOB,
                duration_ms INTEGER NOT NULL,
                sample_rate INTEGER NOT NULL,
                fundamental_frequency REAL NOT NULL,
                formant_frequencies TEXT NOT NULL,
                mfcc TEXT NOT NULL,
                quality_score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES voice_signatures(user_id)
            )",
                [],
            )?;

            // Indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_voice_samples_user ON voice_samples(user_id)",
                [],
            )?;

            conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_samples_timestamp ON voice_samples(timestamp)",
            [],
        )?;
        }

        Ok(Self { db })
    }

    /// Store voice signature
    pub fn store_signature(&self, signature: &VoiceSignature) -> Result<()> {
        let conn = self.db.lock().unwrap();

        // Check if signature exists and is immutable
        let existing: Option<i32> = conn
            .query_row(
                "SELECT immutable FROM voice_signatures WHERE user_id = ?1",
                params![&signature.user_id],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(is_immutable) = existing {
            if is_immutable == 1 {
                return Err(VoiceStoreError::ImmutableSignature(
                    signature.user_id.clone(),
                ));
            }
        }

        let formant_json = serde_json::to_string(&signature.formant_frequencies)?;
        let mfcc_json = serde_json::to_string(&signature.mfcc_signature)?;
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
            "INSERT OR REPLACE INTO voice_signatures (
                user_id, fundamental_frequency, frequency_range_min, frequency_range_max,
                formant_frequencies, speech_rate, mfcc_signature, spectral_centroid,
                jitter, shimmer, harmonic_to_noise_ratio, sample_count, last_updated,
                metadata, tags, immutable
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
            params![
                signature.user_id,
                signature.fundamental_frequency,
                signature.frequency_range.0,
                signature.frequency_range.1,
                formant_json,
                signature.speech_rate,
                mfcc_json,
                signature.spectral_centroid,
                signature.jitter,
                signature.shimmer,
                signature.harmonic_to_noise_ratio,
                signature.sample_count as i64,
                signature.last_updated.to_rfc3339(),
                metadata_json,
                tags_json,
                if signature.immutable { 1 } else { 0 },
            ],
        )?;

        Ok(())
    }

    /// Get voice signature for user
    pub fn get_signature(&self, user_id: &str) -> Result<Option<VoiceSignature>> {
        let conn = self.db.lock().unwrap();

        conn.query_row(
            "SELECT user_id, fundamental_frequency, frequency_range_min, frequency_range_max,
                    formant_frequencies, speech_rate, mfcc_signature, spectral_centroid,
                    jitter, shimmer, harmonic_to_noise_ratio, sample_count, last_updated,
                    metadata, tags, immutable
             FROM voice_signatures WHERE user_id = ?1",
            params![user_id],
            |row| Self::signature_from_row(row),
        )
        .optional()
        .map_err(VoiceStoreError::from)
    }

    /// Get all voice signatures
    pub fn get_all_signatures(&self) -> Result<HashMap<String, VoiceSignature>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT user_id, fundamental_frequency, frequency_range_min, frequency_range_max,
                    formant_frequencies, speech_rate, mfcc_signature, spectral_centroid,
                    jitter, shimmer, harmonic_to_noise_ratio, sample_count, last_updated,
                    metadata, tags, immutable
             FROM voice_signatures",
        )?;

        let signatures = stmt
            .query_map([], |row| Self::signature_from_row(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(signatures
            .into_iter()
            .map(|sig| (sig.user_id.clone(), sig))
            .collect())
    }

    /// Store voice sample
    pub fn store_sample(&self, sample: &VoiceSample) -> Result<()> {
        let conn = self.db.lock().unwrap();

        let formant_json = serde_json::to_string(&sample.formant_frequencies)?;
        let mfcc_json = serde_json::to_string(&sample.mfcc)?;

        conn.execute(
            "INSERT INTO voice_samples (
                id, user_id, audio_data, duration_ms, sample_rate,
                fundamental_frequency, formant_frequencies, mfcc,
                quality_score, timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                sample.id,
                sample.user_id,
                if sample.audio_data.is_empty() {
                    None
                } else {
                    Some(&sample.audio_data)
                },
                sample.duration_ms as i64,
                sample.sample_rate as i64,
                sample.fundamental_frequency,
                formant_json,
                mfcc_json,
                sample.quality_score,
                sample.timestamp.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Get voice samples for user
    pub fn get_samples(&self, user_id: &str, limit: Option<usize>) -> Result<Vec<VoiceSample>> {
        let conn = self.db.lock().unwrap();
        let sql = if let Some(limit) = limit {
            format!(
                "SELECT id, user_id, audio_data, duration_ms, sample_rate,
                        fundamental_frequency, formant_frequencies, mfcc,
                        quality_score, timestamp
                 FROM voice_samples WHERE user_id = ?1
                 ORDER BY timestamp DESC LIMIT {}",
                limit
            )
        } else {
            "SELECT id, user_id, audio_data, duration_ms, sample_rate,
                    fundamental_frequency, formant_frequencies, mfcc,
                    quality_score, timestamp
             FROM voice_samples WHERE user_id = ?1
             ORDER BY timestamp DESC"
                .to_string()
        };

        let mut stmt = conn.prepare(&sql)?;
        let samples = stmt
            .query_map(params![user_id], |row| Self::sample_from_row(row))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(samples)
    }

    /// Delete voice signature
    pub fn delete_signature(&self, user_id: &str) -> Result<()> {
        let conn = self.db.lock().unwrap();

        // Delete samples first (foreign key)
        conn.execute(
            "DELETE FROM voice_samples WHERE user_id = ?1",
            params![user_id],
        )?;

        // Delete signature
        conn.execute(
            "DELETE FROM voice_signatures WHERE user_id = ?1",
            params![user_id],
        )?;

        Ok(())
    }

    /// Count voice signatures
    pub fn count_signatures(&self) -> Result<usize> {
        let conn = self.db.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM voice_signatures", [], |row| {
            row.get(0)
        })?;
        Ok(count as usize)
    }

    /// Count voice samples for user
    pub fn count_samples(&self, user_id: &str) -> Result<usize> {
        let conn = self.db.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM voice_samples WHERE user_id = ?1",
            params![user_id],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    // Helper: Parse signature from row
    fn signature_from_row(row: &Row) -> rusqlite::Result<VoiceSignature> {
        let formant_json: String = row.get(4)?;
        let mfcc_json: String = row.get(6)?;
        let last_updated_str: String = row.get(12)?;
        let metadata_json: Option<String> = row.get(13)?;
        let tags_json: Option<String> = row.get(14)?;
        let immutable: i32 = row.get(15)?;

        let formant_frequencies: Vec<f32> = serde_json::from_str(&formant_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
        })?;

        let mfcc_signature: Vec<f32> = serde_json::from_str(&mfcc_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e))
        })?;

        let last_updated = DateTime::parse_from_rfc3339(&last_updated_str)
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    12,
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
                    13,
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
                    14,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;

        Ok(VoiceSignature {
            user_id: row.get(0)?,
            fundamental_frequency: row.get(1)?,
            frequency_range: (row.get(2)?, row.get(3)?),
            formant_frequencies,
            speech_rate: row.get(5)?,
            mfcc_signature,
            spectral_centroid: row.get(7)?,
            jitter: row.get(8)?,
            shimmer: row.get(9)?,
            harmonic_to_noise_ratio: row.get(10)?,
            sample_count: row.get::<_, i64>(11)? as usize,
            last_updated,
            metadata,
            tags,
            immutable: immutable != 0,
        })
    }

    // Helper: Parse sample from row
    fn sample_from_row(row: &Row) -> rusqlite::Result<VoiceSample> {
        let formant_json: String = row.get(6)?;
        let mfcc_json: String = row.get(7)?;
        let timestamp_str: String = row.get(9)?;

        let formant_frequencies: Vec<f32> = serde_json::from_str(&formant_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(6, rusqlite::types::Type::Text, Box::new(e))
        })?;

        let mfcc: Vec<f32> = serde_json::from_str(&mfcc_json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(7, rusqlite::types::Type::Text, Box::new(e))
        })?;

        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    9,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?
            .with_timezone(&Utc);

        Ok(VoiceSample {
            id: row.get(0)?,
            user_id: row.get(1)?,
            audio_data: row.get::<_, Option<Vec<u8>>>(2)?.unwrap_or_default(),
            duration_ms: row.get::<_, i64>(3)? as u32,
            sample_rate: row.get::<_, i64>(4)? as u32,
            fundamental_frequency: row.get(5)?,
            formant_frequencies,
            mfcc,
            quality_score: row.get(8)?,
            timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signature(user_id: &str) -> VoiceSignature {
        VoiceSignature {
            user_id: user_id.to_string(),
            fundamental_frequency: 150.0,
            frequency_range: (100.0, 300.0),
            formant_frequencies: vec![800.0, 1200.0, 2500.0],
            speech_rate: 4.5,
            mfcc_signature: vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
            ],
            spectral_centroid: 1500.0,
            jitter: 0.5,
            shimmer: 3.0,
            harmonic_to_noise_ratio: 20.0,
            sample_count: 10,
            last_updated: Utc::now(),
            metadata: None,
            tags: None,
            immutable: false,
        }
    }

    fn create_test_sample(user_id: &str) -> VoiceSample {
        VoiceSample {
            id: uuid::Uuid::new_v4().to_string(),
            user_id: user_id.to_string(),
            audio_data: vec![],
            duration_ms: 2000,
            sample_rate: 16000,
            fundamental_frequency: 150.0,
            formant_frequencies: vec![800.0, 1200.0, 2500.0],
            mfcc: vec![
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
            ],
            quality_score: 0.9,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_voice_store_creation() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();
        assert_eq!(store.count_signatures().unwrap(), 0);
    }

    #[test]
    fn test_store_and_retrieve_signature() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        let signature = create_test_signature("magnus");
        store.store_signature(&signature).unwrap();

        let retrieved = store.get_signature("magnus").unwrap().unwrap();
        assert_eq!(retrieved.user_id, "magnus");
        assert_eq!(retrieved.fundamental_frequency, 150.0);
        assert_eq!(retrieved.formant_frequencies.len(), 3);
        assert_eq!(retrieved.mfcc_signature.len(), 13);
        assert_eq!(retrieved.sample_count, 10);
    }

    #[test]
    fn test_update_signature() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        let mut signature = create_test_signature("magnus");
        store.store_signature(&signature).unwrap();

        // Update
        signature.sample_count = 20;
        signature.fundamental_frequency = 155.0;
        store.store_signature(&signature).unwrap();

        let retrieved = store.get_signature("magnus").unwrap().unwrap();
        assert_eq!(retrieved.sample_count, 20);
        assert_eq!(retrieved.fundamental_frequency, 155.0);
    }

    #[test]
    fn test_get_all_signatures() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        store
            .store_signature(&create_test_signature("magnus"))
            .unwrap();
        store
            .store_signature(&create_test_signature("alex"))
            .unwrap();

        let all = store.get_all_signatures().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("magnus"));
        assert!(all.contains_key("alex"));
    }

    #[test]
    fn test_store_and_retrieve_sample() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        // Need signature first (foreign key)
        store
            .store_signature(&create_test_signature("magnus"))
            .unwrap();

        let sample = create_test_sample("magnus");
        store.store_sample(&sample).unwrap();

        let samples = store.get_samples("magnus", None).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].user_id, "magnus");
        assert_eq!(samples[0].duration_ms, 2000);
        assert_eq!(samples[0].sample_rate, 16000);
    }

    #[test]
    fn test_sample_limit() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        store
            .store_signature(&create_test_signature("magnus"))
            .unwrap();

        // Store 5 samples
        for _ in 0..5 {
            store.store_sample(&create_test_sample("magnus")).unwrap();
        }

        let all_samples = store.get_samples("magnus", None).unwrap();
        assert_eq!(all_samples.len(), 5);

        let limited_samples = store.get_samples("magnus", Some(3)).unwrap();
        assert_eq!(limited_samples.len(), 3);
    }

    #[test]
    fn test_delete_signature() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        store
            .store_signature(&create_test_signature("magnus"))
            .unwrap();
        store.store_sample(&create_test_sample("magnus")).unwrap();

        assert_eq!(store.count_signatures().unwrap(), 1);
        assert_eq!(store.count_samples("magnus").unwrap(), 1);

        store.delete_signature("magnus").unwrap();

        assert_eq!(store.count_signatures().unwrap(), 0);
        assert_eq!(store.count_samples("magnus").unwrap(), 0);
    }

    #[test]
    fn test_count_operations() {
        let db = Database::new(":memory:").unwrap();
        let store = VoiceStore::new(&db).unwrap();

        assert_eq!(store.count_signatures().unwrap(), 0);

        store
            .store_signature(&create_test_signature("magnus"))
            .unwrap();
        store
            .store_signature(&create_test_signature("alex"))
            .unwrap();

        assert_eq!(store.count_signatures().unwrap(), 2);

        for _ in 0..3 {
            store.store_sample(&create_test_sample("magnus")).unwrap();
        }

        assert_eq!(store.count_samples("magnus").unwrap(), 3);
        assert_eq!(store.count_samples("alex").unwrap(), 0);
    }
}
