//! Typing Trust Bridge
//!
//! Connects typing pattern recognition to multidimensional trust system.
//! Analyzes linguistic patterns, punctuation style, and behavioral fingerprints.

use sam_vector::database::Database;
use sam_trust::multidimensional_trust::{MultiDimensionalTrustManager, TrustDimension};
use crate::typing_patterns::{TypingPattern, TypingPatternStore};
use anyhow::Result;
use tracing::{debug, info, warn};

/// Typing trust integration
pub struct TypingTrustBridge {
    trust_manager: MultiDimensionalTrustManager,
    pattern_store: TypingPatternStore,
}

impl TypingTrustBridge {
    /// Create new typing trust bridge
    pub fn new(db_path: &str) -> Result<Self> {
        let trust_manager = MultiDimensionalTrustManager::new(db_path)?;
        let database = Database::new(db_path)?;
        let pattern_store = TypingPatternStore::new(&database);
        pattern_store.initialize_schema()?;

        Ok(Self {
            trust_manager,
            pattern_store,
        })
    }

    /// Update typing trust based on pattern similarity
    ///
    /// # Arguments
    /// * `user_id` - User being authenticated
    /// * `similarity` - Typing pattern similarity (0.0-1.0)
    /// * `sample_count` - Number of samples in baseline pattern
    ///
    /// # Returns
    /// * Updated typing trust score
    pub fn update_typing_trust(
        &self,
        user_id: &str,
        similarity: f32,
        sample_count: u32,
    ) -> Result<f64> {
        // Convert similarity to trust score with confidence adjustment
        // More samples = more confident baseline = higher trust ceiling

        let confidence_multiplier = if sample_count >= 10 {
            1.0 // Full confidence with 10+ samples
        } else if sample_count >= 5 {
            0.9 // High confidence with 5-9 samples
        } else if sample_count >= 3 {
            0.75 // Medium confidence with 3-4 samples
        } else {
            0.5 // Low confidence with <3 samples
        };

        let typing_trust = if similarity >= 0.85 {
            // Very high similarity (0.85+) → 0.90-1.0 trust
            0.90 + (similarity - 0.85) as f64 * 0.667
        } else if similarity >= 0.70 {
            // High similarity (0.70-0.85) → 0.75-0.90 trust
            0.75 + (similarity - 0.70) as f64 * 1.0
        } else if similarity >= 0.55 {
            // Medium similarity (0.55-0.70) → 0.50-0.75 trust
            0.50 + (similarity - 0.55) as f64 * 1.667
        } else if similarity >= 0.40 {
            // Low similarity (0.40-0.55) → 0.25-0.50 trust
            0.25 + (similarity - 0.40) as f64 * 1.667
        } else {
            // Very low similarity (<0.40) → 0.0-0.25 trust
            (similarity as f64 * 0.625).max(0.0)
        };

        // Apply confidence multiplier
        let final_trust = (typing_trust * confidence_multiplier).clamp(0.0, 1.0);

        // Update trust dimension
        self.trust_manager
            .update_dimension(user_id, TrustDimension::Typing, final_trust)?;

        debug!(
            user_id,
            similarity,
            sample_count,
            typing_trust = final_trust,
            "Updated typing trust"
        );

        Ok(final_trust)
    }

    /// Process typing authentication event
    ///
    /// Complete workflow: extract patterns, match against baseline, update trust
    ///
    /// # Arguments
    /// * `user_id` - User being authenticated
    /// * `text` - Text sample to analyze
    ///
    /// # Returns
    /// * Typing authentication result with similarity and trust
    pub fn authenticate_typing(&self, user_id: &str, text: &str) -> Result<TypingAuthResult> {
        // Get or create baseline pattern
        let mut baseline = self
            .pattern_store
            .get_pattern(user_id)?
            .unwrap_or_else(|| TypingPattern::new(user_id.to_string()));

        let sample_count = baseline.sample_count;

        // First authentication - establish baseline
        if sample_count == 0 {
            info!(
                user_id,
                "No baseline - establishing initial typing signature"
            );

            // Update baseline with first sample
            baseline.update_with_text(text);
            self.pattern_store.store_pattern(&baseline)?;

            // Low initial trust until more samples
            self.trust_manager
                .update_dimension(user_id, TrustDimension::Typing, 0.1)?;

            return Ok(TypingAuthResult {
                similarity: 0.0,
                typing_trust: 0.1,
                sample_count: 1,
                needs_more_samples: true,
            });
        }

        // Calculate similarity to baseline
        let similarity = baseline.similarity(text);

        // Update trust based on similarity
        let typing_trust = self.update_typing_trust(user_id, similarity, sample_count)?;

        // Incrementally update baseline if similarity is reasonable
        if similarity >= 0.40 {
            baseline.update_with_text(text);
            self.pattern_store.store_pattern(&baseline)?;

            debug!(
                user_id,
                similarity, "Updated typing baseline (incremental learning)"
            );
        } else {
            warn!(
                user_id,
                similarity, "Low similarity - not updating baseline (potential impersonation)"
            );
        }

        Ok(TypingAuthResult {
            similarity,
            typing_trust,
            sample_count: baseline.sample_count,
            needs_more_samples: sample_count < 5,
        })
    }

    /// Identify user by typing pattern (multi-user matching)
    ///
    /// # Arguments
    /// * `text` - Text sample to analyze
    /// * `min_similarity` - Minimum similarity threshold (0.0-1.0)
    ///
    /// # Returns
    /// * List of (user_id, similarity, trust_score) sorted by similarity
    pub fn identify_by_typing(
        &self,
        text: &str,
        min_similarity: f32,
    ) -> Result<Vec<(String, f32, f64)>> {
        // Get all candidate matches from pattern store
        let matches = self
            .pattern_store
            .identify_by_typing(text, min_similarity)?;

        let mut results = Vec::new();

        for (user_id, similarity) in matches {
            // Get pattern to check sample count
            if let Some(pattern) = self.pattern_store.get_pattern(&user_id)? {
                // Calculate what trust score this would yield
                let trust_score =
                    self.calculate_trust_for_similarity(similarity, pattern.sample_count);

                results.push((user_id, similarity, trust_score));
            }
        }

        // Already sorted by similarity from pattern_store
        Ok(results)
    }

    /// Calculate trust score for a given similarity (without updating)
    fn calculate_trust_for_similarity(&self, similarity: f32, sample_count: u32) -> f64 {
        let confidence_multiplier = if sample_count >= 10 {
            1.0
        } else if sample_count >= 5 {
            0.9
        } else if sample_count >= 3 {
            0.75
        } else {
            0.5
        };

        let typing_trust = if similarity >= 0.85 {
            0.90 + (similarity - 0.85) as f64 * 0.667
        } else if similarity >= 0.70 {
            0.75 + (similarity - 0.70) as f64 * 1.0
        } else if similarity >= 0.55 {
            0.50 + (similarity - 0.55) as f64 * 1.667
        } else if similarity >= 0.40 {
            0.25 + (similarity - 0.40) as f64 * 1.667
        } else {
            (similarity as f64 * 0.625).max(0.0)
        };

        (typing_trust * confidence_multiplier).clamp(0.0, 1.0)
    }

    /// Get current typing trust for user
    pub fn get_typing_trust(&self, user_id: &str) -> Result<f64> {
        let dimensions = self.trust_manager.get_trust(user_id)?;
        Ok(dimensions.typing_trust)
    }

    /// Get typing pattern sample count for user
    pub fn get_sample_count(&self, user_id: &str) -> Result<u32> {
        if let Some(pattern) = self.pattern_store.get_pattern(user_id)? {
            Ok(pattern.sample_count)
        } else {
            Ok(0)
        }
    }
}

/// Typing authentication result
#[derive(Debug, Clone)]
pub struct TypingAuthResult {
    /// Typing pattern similarity (0.0-1.0)
    pub similarity: f32,
    /// Updated typing trust score (0.0-1.0)
    pub typing_trust: f64,
    /// Number of samples in baseline
    pub sample_count: u32,
    /// Whether more samples needed for confident baseline
    pub needs_more_samples: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_authentication() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        let text = "Hello, this is a technical test with async code and proper syntax.";
        let result = bridge.authenticate_typing("magnus", text).unwrap();

        // First auth should establish baseline
        assert_eq!(result.similarity, 0.0);
        assert_eq!(result.typing_trust, 0.1); // Low initial trust
        assert_eq!(result.sample_count, 1);
        assert!(result.needs_more_samples);
    }

    #[test]
    fn test_subsequent_authentication_high_similarity() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Establish baseline with technical, formal style
        bridge
            .authenticate_typing(
                "magnus",
                "The implementation uses async functions with await syntax.",
            )
            .unwrap();
        bridge
            .authenticate_typing(
                "magnus",
                "We need to implement the trait for the struct definition.",
            )
            .unwrap();
        bridge
            .authenticate_typing(
                "magnus",
                "The algorithm processes data efficiently with O(n) complexity.",
            )
            .unwrap();

        // Similar text should yield high trust
        let result = bridge
            .authenticate_typing(
                "magnus",
                "The function implements async trait methods correctly.",
            )
            .unwrap();

        assert!(
            result.similarity >= 0.50,
            "Similar style should have high similarity"
        );
        assert!(
            result.typing_trust >= 0.40,
            "High similarity should yield decent trust"
        );
        assert!(result.sample_count >= 3);
    }

    #[test]
    fn test_subsequent_authentication_low_similarity() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Establish baseline with technical, formal style
        bridge
            .authenticate_typing(
                "magnus",
                "The implementation uses async functions with proper syntax.",
            )
            .unwrap();
        bridge
            .authenticate_typing(
                "magnus",
                "We should refactor the database schema for performance.",
            )
            .unwrap();
        bridge
            .authenticate_typing(
                "magnus",
                "The algorithm complexity is acceptable for this use case.",
            )
            .unwrap();

        // Very different style should yield low trust
        let result = bridge
            .authenticate_typing("magnus", "hey!!! whats up lol omg so cool!!!")
            .unwrap();

        assert!(
            result.similarity < 0.50,
            "Different style should have low similarity"
        );
        assert!(
            result.typing_trust < 0.60,
            "Low similarity should yield low trust"
        );
    }

    #[test]
    fn test_confidence_multiplier() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Build up samples
        let technical_samples = vec![
            "The async implementation requires proper error handling.",
            "We need to optimize the database query performance.",
            "The trait bounds ensure type safety at compile time.",
            "Let's refactor this module for better maintainability.",
            "The algorithm uses dynamic programming for efficiency.",
            "We should add comprehensive test coverage here.",
            "The API design follows REST principles correctly.",
            "This function needs better documentation comments.",
            "The concurrency model uses message passing paradigm.",
            "Performance profiling shows acceptable latency metrics.",
        ];

        for sample in &technical_samples {
            bridge.authenticate_typing("magnus", sample).unwrap();
        }

        // With 10+ samples, confidence multiplier should be 1.0
        let result = bridge
            .authenticate_typing(
                "magnus",
                "The implementation handles errors properly with Result types.",
            )
            .unwrap();

        assert_eq!(result.sample_count, 11);
        assert!(!result.needs_more_samples);
        // High sample count + good similarity should yield high trust
        assert!(result.typing_trust >= 0.50);
    }

    #[test]
    fn test_identify_by_typing() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Create distinct patterns for two users
        let magnus_samples = vec![
            "The async implementation uses trait bounds with generic types.",
            "We should refactor the database schema for better performance.",
            "The algorithm complexity is O(n log n) which is acceptable.",
        ];

        let alice_samples = vec![
            "hey how are you doing today?",
            "lol that's so funny omg!!!",
            "cant wait for the weekend woohoo!",
        ];

        for sample in &magnus_samples {
            bridge.authenticate_typing("magnus", sample).unwrap();
        }

        for sample in &alice_samples {
            bridge.authenticate_typing("alice", sample).unwrap();
        }

        // Technical text should match Magnus
        let matches = bridge
            .identify_by_typing(
                "Let's implement the async trait with proper error handling.",
                0.30,
            )
            .unwrap();

        assert!(!matches.is_empty());
        assert_eq!(matches[0].0, "magnus");
        assert!(matches[0].1 >= 0.30); // similarity
        assert!(matches[0].2 >= 0.20); // trust score
    }

    #[test]
    fn test_trust_score_calculation() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Test trust score tiers
        let trust_high = bridge.calculate_trust_for_similarity(0.90, 10);
        assert!(
            trust_high >= 0.90,
            "Very high similarity should yield very high trust"
        );

        let trust_medium = bridge.calculate_trust_for_similarity(0.70, 10);
        assert!(
            trust_medium >= 0.75 && trust_medium < 0.90,
            "Medium-high similarity should yield medium trust"
        );

        let trust_low = bridge.calculate_trust_for_similarity(0.30, 10);
        assert!(trust_low < 0.40, "Low similarity should yield low trust");

        // Test confidence multiplier effect
        let trust_few_samples = bridge.calculate_trust_for_similarity(0.90, 2);
        let trust_many_samples = bridge.calculate_trust_for_similarity(0.90, 10);
        assert!(
            trust_many_samples > trust_few_samples,
            "More samples should yield higher trust for same similarity"
        );
    }

    #[test]
    fn test_get_typing_trust() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Initial trust should be 0.0 (no data)
        let initial_trust = bridge.get_typing_trust("magnus").unwrap();
        assert_eq!(initial_trust, 0.0);

        // After authentication, trust should be updated
        bridge
            .authenticate_typing("magnus", "Technical test message with proper syntax.")
            .unwrap();

        let updated_trust = bridge.get_typing_trust("magnus").unwrap();
        assert!(updated_trust >= 0.1);
    }

    #[test]
    fn test_get_sample_count() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Initial count should be 0
        let initial_count = bridge.get_sample_count("magnus").unwrap();
        assert_eq!(initial_count, 0);

        // After authentication, count should increment
        bridge
            .authenticate_typing("magnus", "Test message one.")
            .unwrap();
        bridge
            .authenticate_typing("magnus", "Test message two.")
            .unwrap();

        let updated_count = bridge.get_sample_count("magnus").unwrap();
        assert_eq!(updated_count, 2);
    }

    #[test]
    fn test_baseline_not_updated_on_low_similarity() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Establish baseline
        bridge
            .authenticate_typing("magnus", "Technical async implementation with traits.")
            .unwrap();
        bridge
            .authenticate_typing("magnus", "Database schema optimization for performance.")
            .unwrap();
        bridge
            .authenticate_typing("magnus", "Algorithm complexity analysis shows efficiency.")
            .unwrap();

        let count_before = bridge.get_sample_count("magnus").unwrap();

        // Very different style - should not update baseline
        bridge
            .authenticate_typing("magnus", "hey lol omg so cool!!!")
            .unwrap();

        let count_after = bridge.get_sample_count("magnus").unwrap();

        // Sample count should not increase if similarity < 0.40
        assert_eq!(count_before, count_after);
    }

    #[test]
    fn test_incremental_learning() {
        let bridge = TypingTrustBridge::new(":memory:").unwrap();

        // Establish baseline
        bridge
            .authenticate_typing("magnus", "Technical async implementation.")
            .unwrap();
        bridge
            .authenticate_typing("magnus", "Database optimization techniques.")
            .unwrap();

        let count_before = bridge.get_sample_count("magnus").unwrap();

        // Similar style should update baseline
        let result = bridge
            .authenticate_typing("magnus", "Algorithm complexity analysis.")
            .unwrap();

        let count_after = bridge.get_sample_count("magnus").unwrap();

        // Should increment if similarity >= 0.40
        if result.similarity >= 0.40 {
            assert_eq!(count_after, count_before + 1);
        }
    }
}
