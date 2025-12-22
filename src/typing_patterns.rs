//! Typing pattern recognition for behavioral fingerprinting
//!
//! Learns how users type to enable identification by typing style,
//! keystroke dynamics, and linguistic patterns.

use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use frame_catalog::database::{Database, DatabaseError, Result};

/// Typing pattern fingerprint for a user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingPattern {
    pub user_id: String,

    // Linguistic fingerprint
    pub common_words: HashMap<String, u32>,
    pub common_bigrams: HashMap<String, u32>,
    pub common_trigrams: HashMap<String, u32>,

    // Punctuation style
    pub comma_frequency: f32,
    pub period_frequency: f32,
    pub exclamation_frequency: f32,
    pub question_frequency: f32,
    pub emoji_frequency: f32,
    pub ellipsis_frequency: f32,

    // Structural patterns
    pub avg_sentence_length: f32,
    pub avg_word_length: f32,
    pub avg_message_length: f32,

    // Capitalization style
    pub capitalization_frequency: f32,
    pub all_caps_frequency: f32,

    // Technical markers
    pub code_block_frequency: f32,
    pub technical_terms_frequency: f32,

    // Semantic patterns
    pub formality_score: f32, // 0.0 = casual, 1.0 = formal

    // Metadata
    pub sample_count: u32,
    pub total_characters: u64,
    pub last_updated: DateTime<Utc>,
}

impl TypingPattern {
    /// Create a new empty typing pattern
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            common_words: HashMap::new(),
            common_bigrams: HashMap::new(),
            common_trigrams: HashMap::new(),
            comma_frequency: 0.0,
            period_frequency: 0.0,
            exclamation_frequency: 0.0,
            question_frequency: 0.0,
            emoji_frequency: 0.0,
            ellipsis_frequency: 0.0,
            avg_sentence_length: 0.0,
            avg_word_length: 0.0,
            avg_message_length: 0.0,
            capitalization_frequency: 0.0,
            all_caps_frequency: 0.0,
            code_block_frequency: 0.0,
            technical_terms_frequency: 0.0,
            formality_score: 0.5,
            sample_count: 0,
            total_characters: 0,
            last_updated: Utc::now(),
        }
    }

    /// Extract patterns from text sample
    pub fn extract_from_text(text: &str) -> PatternFeatures {
        let text_lower = text.to_lowercase();
        let char_count = text.len();
        let word_count = text.split_whitespace().count();

        // Word frequency
        let words: Vec<String> = text_lower
            .split_whitespace()
            .filter(|w| w.len() > 2) // Filter very short words
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect();

        let mut word_freq: HashMap<String, u32> = HashMap::new();
        for word in &words {
            *word_freq.entry(word.clone()).or_insert(0) += 1;
        }

        // Bigrams
        let mut bigrams: HashMap<String, u32> = HashMap::new();
        for window in words.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            *bigrams.entry(bigram).or_insert(0) += 1;
        }

        // Trigrams
        let mut trigrams: HashMap<String, u32> = HashMap::new();
        for window in words.windows(3) {
            let trigram = format!("{} {} {}", window[0], window[1], window[2]);
            *trigrams.entry(trigram).or_insert(0) += 1;
        }

        // Punctuation counts
        let comma_count = text.matches(',').count() as f32;
        let period_count = text.matches('.').count() as f32;
        let exclamation_count = text.matches('!').count() as f32;
        let question_count = text.matches('?').count() as f32;
        let ellipsis_count = text.matches("...").count() as f32;

        // Emoji detection (simplified - check for common emoji characters)
        let emoji_count = text
            .chars()
            .filter(|c| {
                let code = *c as u32;
                (0x1F600..=0x1F64F).contains(&code) || // Emoticons
                (0x1F300..=0x1F5FF).contains(&code) || // Misc Symbols
                (0x1F680..=0x1F6FF).contains(&code) || // Transport
                (0x2600..=0x26FF).contains(&code) // Misc symbols
            })
            .count() as f32;

        // Sentence detection
        let sentence_count = (period_count + exclamation_count + question_count).max(1.0);
        let avg_sentence_length = word_count as f32 / sentence_count;

        // Word length
        let total_word_length: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = if !words.is_empty() {
            total_word_length as f32 / words.len() as f32
        } else {
            0.0
        };

        // Capitalization
        let capital_count = text.chars().filter(|c| c.is_uppercase()).count() as f32;
        let all_caps_words = text
            .split_whitespace()
            .filter(|w| w.len() > 1 && w.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()))
            .count() as f32;

        // Code detection
        let code_block_count = text.matches("```").count() as f32;

        // Technical terms (simplified heuristic)
        let technical_terms = [
            "async",
            "await",
            "impl",
            "trait",
            "struct",
            "enum",
            "fn",
            "pub",
            "mod",
            "use",
            "crate",
            "type",
            "const",
            "static",
            "api",
            "database",
            "server",
            "client",
            "protocol",
            "algorithm",
        ];
        let technical_count = technical_terms
            .iter()
            .map(|term| text_lower.matches(term).count())
            .sum::<usize>() as f32;

        // Formality score (simplified)
        let contractions = ["don't", "won't", "can't", "isn't", "aren't", "didn't"];
        let contraction_count = contractions
            .iter()
            .map(|c| text_lower.matches(c).count())
            .sum::<usize>() as f32;

        let formality_score = if word_count > 0 {
            (1.0 - (contraction_count / word_count as f32))
                .max(0.0)
                .min(1.0)
        } else {
            0.5
        };

        PatternFeatures {
            word_freq,
            bigrams,
            trigrams,
            comma_frequency: comma_count / char_count as f32,
            period_frequency: period_count / char_count as f32,
            exclamation_frequency: exclamation_count / char_count as f32,
            question_frequency: question_count / char_count as f32,
            emoji_frequency: emoji_count / char_count as f32,
            ellipsis_frequency: ellipsis_count / char_count as f32,
            avg_sentence_length,
            avg_word_length,
            message_length: char_count,
            capitalization_frequency: capital_count / char_count as f32,
            all_caps_frequency: all_caps_words / word_count.max(1) as f32,
            code_block_frequency: code_block_count / (char_count as f32 / 100.0).max(1.0),
            technical_frequency: technical_count / word_count.max(1) as f32,
            formality_score,
        }
    }

    /// Update pattern with new text sample (incremental learning)
    pub fn update_with_text(&mut self, text: &str) {
        let features = Self::extract_from_text(text);

        let old_weight = self.sample_count as f32;
        let new_weight = 1.0;
        let total_weight = old_weight + new_weight;

        // Update word frequencies
        for (word, count) in features.word_freq {
            *self.common_words.entry(word).or_insert(0) += count;
        }

        // Update bigrams
        for (bigram, count) in features.bigrams {
            *self.common_bigrams.entry(bigram).or_insert(0) += count;
        }

        // Update trigrams
        for (trigram, count) in features.trigrams {
            *self.common_trigrams.entry(trigram).or_insert(0) += count;
        }

        // Update weighted averages
        self.comma_frequency = (self.comma_frequency * old_weight
            + features.comma_frequency * new_weight)
            / total_weight;
        self.period_frequency = (self.period_frequency * old_weight
            + features.period_frequency * new_weight)
            / total_weight;
        self.exclamation_frequency = (self.exclamation_frequency * old_weight
            + features.exclamation_frequency * new_weight)
            / total_weight;
        self.question_frequency = (self.question_frequency * old_weight
            + features.question_frequency * new_weight)
            / total_weight;
        self.emoji_frequency = (self.emoji_frequency * old_weight
            + features.emoji_frequency * new_weight)
            / total_weight;
        self.ellipsis_frequency = (self.ellipsis_frequency * old_weight
            + features.ellipsis_frequency * new_weight)
            / total_weight;
        self.avg_sentence_length = (self.avg_sentence_length * old_weight
            + features.avg_sentence_length * new_weight)
            / total_weight;
        self.avg_word_length = (self.avg_word_length * old_weight
            + features.avg_word_length * new_weight)
            / total_weight;
        self.avg_message_length = (self.avg_message_length * old_weight
            + features.message_length as f32 * new_weight)
            / total_weight;
        self.capitalization_frequency = (self.capitalization_frequency * old_weight
            + features.capitalization_frequency * new_weight)
            / total_weight;
        self.all_caps_frequency = (self.all_caps_frequency * old_weight
            + features.all_caps_frequency * new_weight)
            / total_weight;
        self.code_block_frequency = (self.code_block_frequency * old_weight
            + features.code_block_frequency * new_weight)
            / total_weight;
        self.technical_terms_frequency = (self.technical_terms_frequency * old_weight
            + features.technical_frequency * new_weight)
            / total_weight;
        self.formality_score = (self.formality_score * old_weight
            + features.formality_score * new_weight)
            / total_weight;

        self.sample_count += 1;
        self.total_characters += text.len() as u64;
        self.last_updated = Utc::now();
    }

    /// Calculate similarity between this pattern and new text
    pub fn similarity(&self, text: &str) -> f32 {
        if self.sample_count == 0 {
            return 0.0;
        }

        let features = Self::extract_from_text(text);

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Word overlap (weighted by frequency)
        let word_overlap = self.calculate_word_overlap(&features.word_freq);
        score += word_overlap * 0.3;
        weight_sum += 0.3;

        // Bigram overlap
        let bigram_overlap = self.calculate_bigram_overlap(&features.bigrams);
        score += bigram_overlap * 0.2;
        weight_sum += 0.2;

        // Punctuation similarity
        let punct_sim = 1.0
            - ((self.comma_frequency - features.comma_frequency).abs()
                + (self.period_frequency - features.period_frequency).abs()
                + (self.exclamation_frequency - features.exclamation_frequency).abs()
                + (self.question_frequency - features.question_frequency).abs()
                + (self.emoji_frequency - features.emoji_frequency).abs())
                / 5.0;
        score += punct_sim * 0.2;
        weight_sum += 0.2;

        // Structural similarity
        let struct_sim = 1.0
            - (((self.avg_sentence_length - features.avg_sentence_length).abs() / 20.0).min(1.0)
                + ((self.avg_word_length - features.avg_word_length).abs() / 5.0).min(1.0))
                / 2.0;
        score += struct_sim * 0.15;
        weight_sum += 0.15;

        // Style similarity
        let style_sim = 1.0
            - ((self.capitalization_frequency - features.capitalization_frequency).abs()
                + (self.code_block_frequency - features.code_block_frequency).abs()
                + (self.technical_terms_frequency - features.technical_frequency).abs()
                + (self.formality_score - features.formality_score).abs())
                / 4.0;
        score += style_sim * 0.15;
        weight_sum += 0.15;

        score / weight_sum
    }

    fn calculate_word_overlap(&self, other_words: &HashMap<String, u32>) -> f32 {
        if self.common_words.is_empty() || other_words.is_empty() {
            return 0.0;
        }

        let mut overlap = 0;
        let mut total = 0;

        for (word, _) in other_words {
            if self.common_words.contains_key(word) {
                overlap += 1;
            }
            total += 1;
        }

        if total > 0 {
            overlap as f32 / total as f32
        } else {
            0.0
        }
    }

    fn calculate_bigram_overlap(&self, other_bigrams: &HashMap<String, u32>) -> f32 {
        if self.common_bigrams.is_empty() || other_bigrams.is_empty() {
            return 0.0;
        }

        let mut overlap = 0;
        let mut total = 0;

        for (bigram, _) in other_bigrams {
            if self.common_bigrams.contains_key(bigram) {
                overlap += 1;
            }
            total += 1;
        }

        if total > 0 {
            overlap as f32 / total as f32
        } else {
            0.0
        }
    }
}

/// Extracted features from a text sample
#[derive(Debug)]
struct PatternFeatures {
    word_freq: HashMap<String, u32>,
    bigrams: HashMap<String, u32>,
    trigrams: HashMap<String, u32>,
    comma_frequency: f32,
    period_frequency: f32,
    exclamation_frequency: f32,
    question_frequency: f32,
    emoji_frequency: f32,
    ellipsis_frequency: f32,
    avg_sentence_length: f32,
    avg_word_length: f32,
    message_length: usize,
    capitalization_frequency: f32,
    all_caps_frequency: f32,
    code_block_frequency: f32,
    technical_frequency: f32,
    formality_score: f32,
}

/// Store for managing typing patterns
pub struct TypingPatternStore {
    db: Arc<Mutex<rusqlite::Connection>>,
}

impl TypingPatternStore {
    pub fn new(database: &Database) -> Self {
        Self {
            db: database.conn(),
        }
    }

    /// Initialize typing patterns schema
    pub fn initialize_schema(&self) -> Result<()> {
        let conn = self.db.lock().unwrap();

        conn.execute(
            "CREATE TABLE IF NOT EXISTS typing_patterns (
                user_id TEXT PRIMARY KEY,
                common_words TEXT NOT NULL,
                common_bigrams TEXT NOT NULL,
                common_trigrams TEXT NOT NULL,
                comma_frequency REAL NOT NULL,
                period_frequency REAL NOT NULL,
                exclamation_frequency REAL NOT NULL,
                question_frequency REAL NOT NULL,
                emoji_frequency REAL NOT NULL,
                ellipsis_frequency REAL NOT NULL,
                avg_sentence_length REAL NOT NULL,
                avg_word_length REAL NOT NULL,
                avg_message_length REAL NOT NULL,
                capitalization_frequency REAL NOT NULL,
                all_caps_frequency REAL NOT NULL,
                code_block_frequency REAL NOT NULL,
                technical_terms_frequency REAL NOT NULL,
                formality_score REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                total_characters INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )",
            [],
        )?;

        Ok(())
    }

    /// Store or update typing pattern
    pub fn store_pattern(&self, pattern: &TypingPattern) -> Result<()> {
        let conn = self.db.lock().unwrap();

        conn.execute(
            "INSERT OR REPLACE INTO typing_patterns (
                user_id, common_words, common_bigrams, common_trigrams,
                comma_frequency, period_frequency, exclamation_frequency,
                question_frequency, emoji_frequency, ellipsis_frequency,
                avg_sentence_length, avg_word_length, avg_message_length,
                capitalization_frequency, all_caps_frequency, code_block_frequency,
                technical_terms_frequency, formality_score,
                sample_count, total_characters, last_updated
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            params![
                pattern.user_id,
                serde_json::to_string(&pattern.common_words).map_err(|e| DatabaseError::Serialization(e.to_string()))?,
                serde_json::to_string(&pattern.common_bigrams).map_err(|e| DatabaseError::Serialization(e.to_string()))?,
                serde_json::to_string(&pattern.common_trigrams).map_err(|e| DatabaseError::Serialization(e.to_string()))?,
                pattern.comma_frequency,
                pattern.period_frequency,
                pattern.exclamation_frequency,
                pattern.question_frequency,
                pattern.emoji_frequency,
                pattern.ellipsis_frequency,
                pattern.avg_sentence_length,
                pattern.avg_word_length,
                pattern.avg_message_length,
                pattern.capitalization_frequency,
                pattern.all_caps_frequency,
                pattern.code_block_frequency,
                pattern.technical_terms_frequency,
                pattern.formality_score,
                pattern.sample_count,
                pattern.total_characters as i64,
                pattern.last_updated.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Get typing pattern for user
    pub fn get_pattern(&self, user_id: &str) -> Result<Option<TypingPattern>> {
        let conn = self.db.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT user_id, common_words, common_bigrams, common_trigrams,
                    comma_frequency, period_frequency, exclamation_frequency,
                    question_frequency, emoji_frequency, ellipsis_frequency,
                    avg_sentence_length, avg_word_length, avg_message_length,
                    capitalization_frequency, all_caps_frequency, code_block_frequency,
                    technical_terms_frequency, formality_score,
                    sample_count, total_characters, last_updated
             FROM typing_patterns WHERE user_id = ?1",
                params![user_id],
                |row| {
                    Ok(TypingPattern {
                        user_id: row.get(0)?,
                        common_words: serde_json::from_str(&row.get::<_, String>(1)?)
                            .unwrap_or_default(),
                        common_bigrams: serde_json::from_str(&row.get::<_, String>(2)?)
                            .unwrap_or_default(),
                        common_trigrams: serde_json::from_str(&row.get::<_, String>(3)?)
                            .unwrap_or_default(),
                        comma_frequency: row.get(4)?,
                        period_frequency: row.get(5)?,
                        exclamation_frequency: row.get(6)?,
                        question_frequency: row.get(7)?,
                        emoji_frequency: row.get(8)?,
                        ellipsis_frequency: row.get(9)?,
                        avg_sentence_length: row.get(10)?,
                        avg_word_length: row.get(11)?,
                        avg_message_length: row.get(12)?,
                        capitalization_frequency: row.get(13)?,
                        all_caps_frequency: row.get(14)?,
                        code_block_frequency: row.get(15)?,
                        technical_terms_frequency: row.get(16)?,
                        formality_score: row.get(17)?,
                        sample_count: row.get(18)?,
                        total_characters: row.get::<_, i64>(19)? as u64,
                        last_updated: DateTime::parse_from_rfc3339(&row.get::<_, String>(20)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Identify user by typing pattern
    pub fn identify_by_typing(
        &self,
        text: &str,
        min_confidence: f32,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.db.lock().unwrap();

        let mut stmt =
            conn.prepare("SELECT user_id FROM typing_patterns WHERE sample_count >= 3")?;

        let user_ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(conn);

        let mut matches = Vec::new();

        for user_id in user_ids {
            if let Some(pattern) = self.get_pattern(&user_id)? {
                let confidence = pattern.similarity(text);
                if confidence >= min_confidence {
                    matches.push((user_id, confidence));
                }
            }
        }

        // Sort by confidence descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_extraction() {
        let text = "Hello! This is a test. How are you doing? I'm writing code...";
        let features = TypingPattern::extract_from_text(text);

        assert!(features.exclamation_frequency > 0.0);
        assert!(features.question_frequency > 0.0);
        assert!(features.ellipsis_frequency > 0.0);
        assert!(features.avg_sentence_length > 0.0);
    }

    #[test]
    fn test_pattern_update() {
        let mut pattern = TypingPattern::new("test-user".to_string());

        pattern.update_with_text("Hello, this is a technical test with async code.");
        pattern.update_with_text("Another message, similar style and technical terms.");

        assert_eq!(pattern.sample_count, 2);
        assert!(pattern.common_words.contains_key("technical"));
        assert!(pattern.technical_terms_frequency > 0.0);
    }

    #[test]
    fn test_pattern_similarity() {
        let mut pattern = TypingPattern::new("test-user".to_string());

        // Train with technical, formal text
        pattern.update_with_text("The implementation uses async functions with await syntax.");
        pattern.update_with_text("We need to implement the trait for the struct.");
        pattern.update_with_text("The algorithm processes data efficiently.");

        // Similar text should have high similarity
        let similar = pattern.similarity("The function implements async trait methods.");
        assert!(
            similar > 0.5,
            "Similar text should have high similarity: {}",
            similar
        );

        // Different style should have lower similarity
        let different = pattern.similarity("hey!!! whats up??? lol omg!!!");
        assert!(
            different < 0.5,
            "Different style should have low similarity: {}",
            different
        );
    }

    #[test]
    fn test_pattern_store() {
        let db = Database::new(":memory:").unwrap();
        let store = TypingPatternStore::new(&db);
        store.initialize_schema().unwrap();

        let mut pattern = TypingPattern::new("test-user".to_string());
        pattern.update_with_text("Test message for pattern storage.");

        store.store_pattern(&pattern).unwrap();

        let retrieved = store.get_pattern("test-user").unwrap().unwrap();
        assert_eq!(retrieved.user_id, "test-user");
        assert_eq!(retrieved.sample_count, 1);
    }

    #[test]
    fn test_identify_by_typing() {
        let db = Database::new(":memory:").unwrap();
        let store = TypingPatternStore::new(&db);
        store.initialize_schema().unwrap();

        // Create pattern for "magnus"
        let mut magnus_pattern = TypingPattern::new("magnus".to_string());
        magnus_pattern
            .update_with_text("The async implementation uses trait bounds with generic types.");
        magnus_pattern
            .update_with_text("We should refactor the database schema for better performance.");
        magnus_pattern
            .update_with_text("The algorithm complexity is O(n log n) which is acceptable.");

        store.store_pattern(&magnus_pattern).unwrap();

        // Similar text should match Magnus
        let matches = store
            .identify_by_typing(
                "Let's implement the async trait with proper error handling.",
                0.4,
            )
            .unwrap();

        assert!(!matches.is_empty());
        assert_eq!(matches[0].0, "magnus");
        assert!(matches[0].1 > 0.4);
    }
}
