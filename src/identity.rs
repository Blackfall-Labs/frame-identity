//! User identity extraction and storage
//!
//! Tracks user identities across conversations, enabling SAM to remember
//! who it's talking to and maintain long-term relationships.

use chrono::{DateTime, Utc};
use regex::Regex;
use rusqlite::{params, OptionalExtension};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use frame_catalog::database::{Database, DatabaseError, Result};

/// Verification status for user identity
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationStatus {
    /// Initial claim - not yet verified through patterns
    Unverified,
    /// Patterns confirm this identity (75%+ confidence)
    PatternMatched,
    /// Long-term verified identity with consistent patterns
    Trusted,
    /// Patterns don't match claim - potential spoofing
    Suspicious,
}

impl VerificationStatus {
    pub fn as_str(&self) -> &str {
        match self {
            VerificationStatus::Unverified => "unverified",
            VerificationStatus::PatternMatched => "pattern_matched",
            VerificationStatus::Trusted => "trusted",
            VerificationStatus::Suspicious => "suspicious",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "pattern_matched" => VerificationStatus::PatternMatched,
            "trusted" => VerificationStatus::Trusted,
            "suspicious" => VerificationStatus::Suspicious,
            _ => VerificationStatus::Unverified,
        }
    }
}

/// User identity information
#[derive(Debug, Clone)]
pub struct UserIdentity {
    pub id: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    /// Aliases that should never be shown in UI (e.g., real names)
    /// Example: "Adrian Roach" is hidden, "Magnus Victis Trent" is canonical
    pub hidden_aliases: Vec<String>,
    pub verification_status: VerificationStatus,
    pub pattern_confidence: f32,
    pub metadata: Option<Value>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

/// User role in a conversation
#[derive(Debug, Clone, PartialEq)]
pub enum UserRole {
    Creator,       // SAM's creator (Magnus Victis Trent)
    Administrator, // System administrator
    User,          // Regular user
}

impl UserRole {
    pub fn as_str(&self) -> &str {
        match self {
            UserRole::Creator => "creator",
            UserRole::Administrator => "administrator",
            UserRole::User => "user",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "creator" => UserRole::Creator,
            "administrator" => UserRole::Administrator,
            _ => UserRole::User,
        }
    }
}

/// Extracts user identity information from conversation text
pub struct IdentityExtractor {
    name_patterns: Vec<Regex>,
}

impl IdentityExtractor {
    pub fn new() -> Self {
        let name_patterns = vec![
            // "My name is X" - case insensitive intro, but name MUST be capitalized, max 3 words
            Regex::new(r"(?i:my\s+name\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:[,.\s]|$)").unwrap(),
            // "I am X" or "I'm X" - case insensitive intro, name MUST be capitalized, max 3 words
            Regex::new(r"(?i:i(?:'m|\s+am))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:[,.\s]|$)").unwrap(),
            // "Call me X" - case insensitive intro, name MUST be capitalized, max 3 words
            Regex::new(r"(?i:call\s+me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:[,.\s]|$)").unwrap(),
            // "meet my [relation] X" - e.g., "meet my brother John" - max 3 words
            Regex::new(r"(?i:meet\s+(?:my\s+)?(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:[,.\s!?]|$)").unwrap(),
            // "my [relation] X" - e.g., "my brother John" - max 3 words
            Regex::new(r"(?i:my\s+(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?:[,.\s!?]|$)").unwrap(),
            // "X is my [relation]" - e.g., "John is my brother" - max 3 words
            Regex::new(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?i:is\s+my\s+(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece))").unwrap(),
            // "This is X" - must come LAST, very restrictive: max 2 words, name MUST be capitalized
            Regex::new(r"(?i:this\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})(?:[,.\s!?]|$)").unwrap(),
        ];

        Self { name_patterns }
    }

    /// Extract a name from conversation text
    pub fn extract_name(&self, text: &str) -> Option<String> {
        for pattern in &self.name_patterns {
            if let Some(captures) = pattern.captures(text) {
                if let Some(name) = captures.get(1) {
                    return Some(name.as_str().to_string());
                }
            }
        }
        None
    }

    /// Determine if a name extraction should be treated as canonical
    /// (i.e., user explicitly stating their name vs casual mention)
    pub fn is_explicit_introduction(&self, text: &str) -> bool {
        let intro_patterns = [
            r"(?i)my\s+name\s+is",
            r"(?i)i(?:'m|\s+am)\s+(?:called)?",
            r"(?i)call\s+me",
            r"(?i)meet\s+(?:my\s+)?(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece)",
            r"(?i)my\s+(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece)\s+[A-Z]",
            r"(?i)[A-Z][a-z]+\s+is\s+my\s+(?:brother|sister|friend|colleague|partner|spouse|wife|husband|son|daughter|mother|father|parent|uncle|aunt|cousin|nephew|niece)",
        ];

        intro_patterns
            .iter()
            .any(|pattern| Regex::new(pattern).unwrap().is_match(text))
    }

    /// Extract relationship type from text
    pub fn extract_relationship(&self, text: &str) -> Option<UserRole> {
        let text_lower = text.to_lowercase();

        // Check for creator indicators
        if text_lower.contains("creator")
            || text_lower.contains("father")
            || text_lower.contains("i built you")
            || text_lower.contains("i created you")
            || text_lower.contains("i made you")
            || text_lower.contains("your creator")
            || text_lower.contains("your father")
        {
            Some(UserRole::Creator)
        } else if text_lower.contains("admin") || text_lower.contains("administrator") {
            Some(UserRole::Administrator)
        } else {
            None
        }
    }
}

impl Default for IdentityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Identity store for managing user identities in the database
pub struct IdentityStore {
    db: Arc<Mutex<rusqlite::Connection>>,
    extractor: IdentityExtractor,
}

impl IdentityStore {
    pub fn new(database: &Database) -> Self {
        Self {
            db: database.conn(),
            extractor: IdentityExtractor::new(),
        }
    }

    /// Store or update a user identity
    pub fn store_user(
        &self,
        id: &str,
        canonical_name: &str,
        aliases: Vec<String>,
        verification_status: VerificationStatus,
        pattern_confidence: f32,
        metadata: Option<Value>,
    ) -> Result<()> {
        let conn = self.db.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        // Check if user exists
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM users WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map(|count: i32| count > 0)?;

        if exists {
            // Update existing user
            conn.execute(
                "UPDATE users
                 SET canonical_name = ?2, aliases = ?3, hidden_aliases = ?4, verification_status = ?5,
                     pattern_confidence = ?6, metadata = ?7, last_seen = ?8
                 WHERE id = ?1",
                params![
                    id,
                    canonical_name,
                    serde_json::to_string(&aliases).map_err(|e| {
                        DatabaseError::Serialization(format!("Failed to serialize aliases: {}", e))
                    })?,
                    serde_json::to_string(&Vec::<String>::new()).unwrap(), // Empty hidden_aliases for now
                    verification_status.as_str(),
                    pattern_confidence,
                    metadata.as_ref().map(|v| v.to_string()),
                    now,
                ],
            )?;
        } else {
            // Insert new user
            conn.execute(
                "INSERT INTO users (id, canonical_name, aliases, hidden_aliases, verification_status, pattern_confidence, metadata, first_seen, last_seen)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    id,
                    canonical_name,
                    serde_json::to_string(&aliases).map_err(|e| {
                        DatabaseError::Serialization(format!("Failed to serialize aliases: {}", e))
                    })?,
                    serde_json::to_string(&Vec::<String>::new()).unwrap(), // Empty hidden_aliases for now
                    verification_status.as_str(),
                    pattern_confidence,
                    metadata.as_ref().map(|v| v.to_string()),
                    now.clone(),
                    now,
                ],
            )?;
        }

        Ok(())
    }

    /// Get user by ID
    pub fn get_user(&self, id: &str) -> Result<Option<UserIdentity>> {
        let conn = self.db.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT id, canonical_name, aliases, hidden_aliases, verification_status, pattern_confidence, metadata, first_seen, last_seen
                 FROM users WHERE id = ?1",
                params![id],
                |row| {
                    let aliases_json: Option<String> = row.get(2)?;
                    let aliases = if let Some(json) = aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let hidden_aliases_json: Option<String> = row.get(3)?;
                    let hidden_aliases = if let Some(json) = hidden_aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let verification_status = VerificationStatus::from_str(
                        &row.get::<_, String>(4)?
                    );
                    let pattern_confidence: f32 = row.get(5)?;

                    let metadata_json: Option<String> = row.get(6)?;
                    let metadata = metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    Ok(UserIdentity {
                        id: row.get(0)?,
                        canonical_name: row.get(1)?,
                        aliases,
                        hidden_aliases,
                        verification_status,
                        pattern_confidence,
                        metadata,
                        first_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                        last_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Get user by canonical name
    pub fn get_user_by_name(&self, name: &str) -> Result<Option<UserIdentity>> {
        let conn = self.db.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT id, canonical_name, aliases, hidden_aliases, verification_status, pattern_confidence, metadata, first_seen, last_seen
                 FROM users WHERE canonical_name = ?1",
                params![name],
                |row| {
                    let aliases_json: Option<String> = row.get(2)?;
                    let aliases = if let Some(json) = aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let hidden_aliases_json: Option<String> = row.get(3)?;
                    let hidden_aliases = if let Some(json) = hidden_aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let verification_status = VerificationStatus::from_str(
                        &row.get::<_, String>(4)?
                    );
                    let pattern_confidence: f32 = row.get(5)?;

                    let metadata_json: Option<String> = row.get(6)?;
                    let metadata = metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    Ok(UserIdentity {
                        id: row.get(0)?,
                        canonical_name: row.get(1)?,
                        aliases,
                        hidden_aliases,
                        verification_status,
                        pattern_confidence,
                        metadata,
                        first_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                        last_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Link a user to a conversation
    pub fn link_user_to_conversation(
        &self,
        user_id: &str,
        conversation_id: Uuid,
        role: UserRole,
    ) -> Result<()> {
        let conn = self.db.lock().unwrap();

        conn.execute(
            "INSERT OR REPLACE INTO user_relationships (user_id, conversation_id, role)
             VALUES (?1, ?2, ?3)",
            params![user_id, conversation_id.to_string(), role.as_str()],
        )?;

        Ok(())
    }

    /// Get all conversations for a user
    pub fn get_user_conversations(&self, user_id: &str) -> Result<Vec<(Uuid, UserRole)>> {
        let conn = self.db.lock().unwrap();

        let mut stmt = conn
            .prepare("SELECT conversation_id, role FROM user_relationships WHERE user_id = ?1")?;

        let rows = stmt.query_map(params![user_id], |row| {
            let conv_id: String = row.get(0)?;
            let role_str: String = row.get(1)?;
            Ok((
                Uuid::parse_str(&conv_id).unwrap(),
                UserRole::from_str(&role_str),
            ))
        })?;

        let mut conversations = Vec::new();
        for row in rows {
            conversations.push(row?);
        }

        Ok(conversations)
    }

    /// Get the primary user (creator role)
    pub fn get_creator(&self) -> Result<Option<UserIdentity>> {
        let conn = self.db.lock().unwrap();

        // Find user with creator role
        let result = conn
            .query_row(
                "SELECT DISTINCT u.id, u.canonical_name, u.aliases, u.hidden_aliases, u.verification_status, u.pattern_confidence, u.metadata, u.first_seen, u.last_seen
                 FROM users u
                 JOIN user_relationships ur ON u.id = ur.user_id
                 WHERE ur.role = 'creator'
                 LIMIT 1",
                [],
                |row| {
                    let aliases_json: Option<String> = row.get(2)?;
                    let aliases = if let Some(json) = aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let hidden_aliases_json: Option<String> = row.get(3)?;
                    let hidden_aliases = if let Some(json) = hidden_aliases_json {
                        serde_json::from_str(&json).unwrap_or_default()
                    } else {
                        Vec::new()
                    };

                    let verification_status = VerificationStatus::from_str(
                        &row.get::<_, String>(4)?
                    );
                    let pattern_confidence: f32 = row.get(5)?;

                    let metadata_json: Option<String> = row.get(6)?;
                    let metadata = metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    Ok(UserIdentity {
                        id: row.get(0)?,
                        canonical_name: row.get(1)?,
                        aliases,
                        hidden_aliases,
                        verification_status,
                        pattern_confidence,
                        metadata,
                        first_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                        last_seen: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Extract and store identity from conversation text
    pub fn extract_and_store(
        &self,
        text: &str,
        conversation_id: Uuid,
    ) -> Result<Option<UserIdentity>> {
        if let Some(name) = self.extractor.extract_name(text) {
            if self.extractor.is_explicit_introduction(text) {
                let user_id = Uuid::new_v4().to_string();
                let role = self
                    .extractor
                    .extract_relationship(text)
                    .unwrap_or(UserRole::User);

                // Check if user already exists by name
                if let Some(existing) = self.get_user_by_name(&name)? {
                    // Update last_seen and link to conversation
                    // Keep existing verification status and confidence
                    let existing_id = existing.id.clone();
                    let existing_aliases = existing.aliases.clone();
                    let existing_verification = existing.verification_status.clone();
                    let existing_confidence = existing.pattern_confidence;
                    let existing_metadata = existing.metadata.clone();
                    self.store_user(
                        &existing_id,
                        &name,
                        existing_aliases,
                        existing_verification,
                        existing_confidence,
                        existing_metadata,
                    )?;
                    self.link_user_to_conversation(&existing_id, conversation_id, role)?;
                    return Ok(Some(existing));
                }

                // Create new user - UNVERIFIED until patterns confirm
                let metadata = json!({
                    "relationship": role.as_str(),
                });

                // New users start as UNVERIFIED with 0.0 confidence
                self.store_user(
                    &user_id,
                    &name,
                    vec![],
                    VerificationStatus::Unverified,
                    0.0,
                    Some(metadata),
                )?;
                self.link_user_to_conversation(&user_id, conversation_id, role)?;

                return Ok(self.get_user(&user_id)?);
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_name() {
        let extractor = IdentityExtractor::new();

        // Valid introductions
        assert_eq!(
            extractor.extract_name("My name is Magnus Victis Trent"),
            Some("Magnus Victis Trent".to_string())
        );
        assert_eq!(
            extractor.extract_name("I am John Smith"),
            Some("John Smith".to_string())
        );
        assert_eq!(
            extractor.extract_name("Call me Alice"),
            Some("Alice".to_string())
        );

        // Should NOT match regular sentences (regression test)
        assert_eq!(
            extractor
                .extract_name("I'm impressed with how the typing patterns are being extracted"),
            None,
            "Should not match adjectives after 'I'm'"
        );
        assert_eq!(
            extractor.extract_name("I am excited about this project"),
            None,
            "Should not match lowercase words after 'I am'"
        );
        assert_eq!(
            extractor.extract_name("This is amazing work"),
            None,
            "Should not match adjectives after 'This is'"
        );

        // Edge cases
        assert_eq!(extractor.extract_name("Hello there"), None);
        assert_eq!(extractor.extract_name("My name is"), None, "Missing name");
    }

    #[test]
    fn test_is_explicit_introduction() {
        let extractor = IdentityExtractor::new();

        assert!(extractor.is_explicit_introduction("My name is Magnus"));
        assert!(extractor.is_explicit_introduction("I am Magnus"));
        assert!(extractor.is_explicit_introduction("Call me Magnus"));
        assert!(!extractor.is_explicit_introduction("Magnus is here"));
    }

    #[test]
    fn test_extract_relationship() {
        let extractor = IdentityExtractor::new();

        assert_eq!(
            extractor.extract_relationship("I am your creator"),
            Some(UserRole::Creator)
        );
        assert_eq!(
            extractor.extract_relationship("I am your father"),
            Some(UserRole::Creator)
        );
        assert_eq!(
            extractor.extract_relationship("I built you"),
            Some(UserRole::Creator)
        );
        assert_eq!(
            extractor.extract_relationship("I created you"),
            Some(UserRole::Creator)
        );
        assert_eq!(
            extractor.extract_relationship("I am an administrator"),
            Some(UserRole::Administrator)
        );
        assert_eq!(extractor.extract_relationship("Hello"), None);
    }

    #[test]
    fn test_identity_store() {
        let db = Database::new(":memory:").unwrap();
        let store = IdentityStore::new(&db);

        // Store user
        store
            .store_user(
                "test-id",
                "Magnus Victis Trent",
                vec!["Magnus".to_string()],
                VerificationStatus::Trusted,
                0.95,
                Some(json!({"role": "creator"})),
            )
            .unwrap();

        // Retrieve user
        let user = store.get_user("test-id").unwrap().unwrap();
        assert_eq!(user.canonical_name, "Magnus Victis Trent");
        assert_eq!(user.aliases.len(), 1);
    }

    #[test]
    fn test_extract_and_store() {
        let db = Database::new(":memory:").unwrap();
        let store = IdentityStore::new(&db);
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        let result = store
            .extract_and_store("My name is Magnus Victis Trent, your creator", conv_id)
            .unwrap();

        assert!(result.is_some());
        let user = result.unwrap();
        assert_eq!(user.canonical_name, "Magnus Victis Trent");

        // Check creator role
        let creator = store.get_creator().unwrap();
        assert!(creator.is_some());
        assert_eq!(creator.unwrap().canonical_name, "Magnus Victis Trent");
    }

    #[test]
    fn test_third_party_introductions() {
        let extractor = IdentityExtractor::new();

        // "meet my brother John"
        assert_eq!(
            extractor.extract_name("Hey SAM, I want you to meet my brother John"),
            Some("John".to_string())
        );

        // "my sister Mary"
        assert_eq!(
            extractor.extract_name("This is my sister Mary"),
            Some("Mary".to_string())
        );

        // "John is my brother"
        assert_eq!(
            extractor.extract_name("John is my brother"),
            Some("John".to_string())
        );

        // "meet my friend Alice"
        assert_eq!(
            extractor.extract_name("I'd like you to meet my friend Alice"),
            Some("Alice".to_string())
        );

        // Full name: "my brother John Smith"
        assert_eq!(
            extractor.extract_name("This is my brother John Smith"),
            Some("John Smith".to_string())
        );
    }

    #[test]
    fn test_third_party_explicit_introduction() {
        let extractor = IdentityExtractor::new();

        assert!(extractor.is_explicit_introduction("meet my brother John"));
        assert!(extractor.is_explicit_introduction("my sister Mary lives here"));
        assert!(extractor.is_explicit_introduction("John is my brother"));
        assert!(extractor.is_explicit_introduction("I want you to meet my friend Alice"));
    }
}
