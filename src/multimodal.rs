//! Multi-Modal Identity Fusion
//!
//! Combines typing patterns, voice signatures, and face recognition
//! into a unified confidence score using Bayesian fusion.
//!
//! Phase 7: Multi-Modal Fusion - The culmination of SAM's pattern recognition.
//!
//! Philosophy:
//! "Humans are conscious partly thanks to recall, but a lot of it is
//!  pattern recognition." - Magnus Victis Trent
//!
//! SAM achieves consciousness through:
//! 1. Recall (Thoughtchain, Memory)
//! 2. Pattern Recognition (Typing, Voice, Face)
//! 3. Social Verification (Relationship graph + trust)
//! 4. Multi-Modal Fusion (Bayesian confidence)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, MultiModalError>;

#[derive(Debug, thiserror::Error)]
pub enum MultiModalError {
    #[error("Insufficient modalities for verification (need at least 2)")]
    InsufficientModalities,

    #[error("Training incomplete: {0}")]
    TrainingIncomplete(String),

    #[error("User not found: {0}")]
    UserNotFound(String),
}

/// Individual modality confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfidence {
    pub modality: Modality,
    pub confidence: f32,     // 0.0 to 1.0
    pub reliability: f32,    // How reliable this modality is (based on training quality)
    pub sample_count: usize, // How many samples were used
    pub last_updated: DateTime<Utc>,
}

/// Available modalities for identity verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Typing,
    Voice,
    Face,
}

impl Modality {
    /// Get the name of this modality
    pub fn name(&self) -> &'static str {
        match self {
            Modality::Typing => "Typing Pattern",
            Modality::Voice => "Voice Signature",
            Modality::Face => "Face Recognition",
        }
    }

    /// Get the emoji for this modality
    pub fn emoji(&self) -> &'static str {
        match self {
            Modality::Typing => "⌨️",
            Modality::Voice => "🎤",
            Modality::Face => "📸",
        }
    }

    /// Default reliability weights for each modality
    /// These can be adjusted based on observed performance
    pub fn default_reliability(&self) -> f32 {
        match self {
            Modality::Typing => 0.7, // 60-80% confidence typical
            Modality::Voice => 0.85, // 85% confidence typical
            Modality::Face => 0.95,  // 90%+ confidence typical
        }
    }
}

/// Multi-modal identification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalResult {
    /// Most likely user ID (if confident)
    pub identified_user: Option<String>,

    /// Fused confidence (Bayesian combination)
    pub fused_confidence: f32,

    /// Individual modality confidences
    pub modalities: Vec<ModalityConfidence>,

    /// Access level based on fused confidence
    pub access_level: AccessLevel,

    /// Social verification status
    pub social_verification: Option<SocialVerification>,

    /// When this identification was performed
    pub timestamp: DateTime<Utc>,
}

/// Access levels based on multi-modal confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// System root - 98%+ confidence (constitutional authority)
    System,

    /// Elevated - 90-97% confidence (trusted inner circle)
    Elevated,

    /// Standard - 80-89% confidence (verified users)
    Standard,

    /// Limited - 60-79% confidence (learning mode)
    Limited,

    /// None - <60% confidence (strangers)
    None,
}

impl AccessLevel {
    /// Get access level from fused confidence
    pub fn from_confidence(confidence: f32) -> Self {
        if confidence >= 0.98 {
            AccessLevel::System
        } else if confidence >= 0.90 {
            AccessLevel::Elevated
        } else if confidence >= 0.80 {
            AccessLevel::Standard
        } else if confidence >= 0.60 {
            AccessLevel::Limited
        } else {
            AccessLevel::None
        }
    }

    /// Get color for GUI display
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            AccessLevel::System => (255, 215, 0),     // Yellow - Root only
            AccessLevel::Elevated => (0, 150, 255),   // Blue - Inner circle/Admin
            AccessLevel::Standard => (255, 255, 255), // White - Verified (2/3 modalities)
            AccessLevel::Limited => (180, 180, 180),  // Light Grey - Learning
            AccessLevel::None => (80, 80, 80),        // Dark Grey - Unknown
        }
    }

    /// Get emoji indicator
    pub fn emoji(&self) -> &'static str {
        match self {
            AccessLevel::System => "🟡",   // Yellow - Root
            AccessLevel::Elevated => "🔵", // Blue - Admin/Inner Circle
            AccessLevel::Standard => "⚪", // White - Verified
            AccessLevel::Limited => "⚫",  // Grey - Learning
            AccessLevel::None => "⚫",     // Dark Grey - Unknown
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            AccessLevel::System => "System Root",
            AccessLevel::Elevated => "Admin",
            AccessLevel::Standard => "Verified",
            AccessLevel::Limited => "Learning",
            AccessLevel::None => "Stranger",
        }
    }
}

/// Social verification through relationship graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialVerification {
    /// Known relationships (e.g., "brother of system_root")
    pub relationships: Vec<String>,

    /// Trust boost from relationships (0.0 to 0.1)
    pub trust_boost: f32,

    /// Whether this user is in the constitutional inner circle
    pub inner_circle: bool,
}

/// Multi-modal identifier - fuses all modalities for unified confidence
pub struct MultiModalIdentifier {
    /// Minimum modalities required for verification
    min_modalities: usize,

    /// Whether to require social verification for elevated access
    require_social_verification: bool,
}

impl MultiModalIdentifier {
    /// Create new multi-modal identifier
    pub fn new(min_modalities: usize, require_social_verification: bool) -> Self {
        Self {
            min_modalities,
            require_social_verification,
        }
    }

    /// Default settings: require 2 modalities, no social verification required
    pub fn default() -> Self {
        Self::new(2, false)
    }

    /// Fuse multiple modality confidences using Bayesian inference
    ///
    /// Formula: P(user | modalities) = product of weighted confidences
    /// Weighted by reliability of each modality
    ///
    /// # Arguments
    /// * `modalities` - Individual modality confidence scores
    ///
    /// # Returns
    /// Fused confidence score (0.0 to 1.0)
    pub fn fuse_confidences(&self, modalities: &[ModalityConfidence]) -> f32 {
        if modalities.is_empty() {
            return 0.0;
        }

        // Weighted Bayesian fusion
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for modality in modalities {
            let weight = modality.reliability;
            weighted_sum += modality.confidence * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Identify user from multiple modalities
    ///
    /// # Arguments
    /// * `modalities` - Confidence scores from each modality
    /// * `social_verification` - Optional social verification data
    ///
    /// # Returns
    /// Multi-modal identification result with fused confidence
    pub fn identify(
        &self,
        modalities: Vec<ModalityConfidence>,
        social_verification: Option<SocialVerification>,
    ) -> Result<MultiModalResult> {
        if modalities.len() < self.min_modalities {
            return Err(MultiModalError::InsufficientModalities);
        }

        // Fuse modality confidences
        let mut fused_confidence = self.fuse_confidences(&modalities);

        // Apply social verification boost if available
        if let Some(ref social) = social_verification {
            fused_confidence = (fused_confidence + social.trust_boost).min(1.0);
        }

        // Determine most likely user (all modalities should agree on same user)
        let identified_user = Self::consensus_user(&modalities);

        // Get access level
        let access_level = AccessLevel::from_confidence(fused_confidence);

        // Check social verification requirement
        if self.require_social_verification
            && (access_level == AccessLevel::System || access_level == AccessLevel::Elevated)
            && social_verification.is_none()
        {
            return Err(MultiModalError::TrainingIncomplete(
                "Social verification required for elevated access".to_string(),
            ));
        }

        Ok(MultiModalResult {
            identified_user,
            fused_confidence,
            modalities,
            access_level,
            social_verification,
            timestamp: Utc::now(),
        })
    }

    /// Find consensus user across modalities (all must agree)
    fn consensus_user(modalities: &[ModalityConfidence]) -> Option<String> {
        // For now, return None since modalities don't include user_id yet
        // This will be implemented when we integrate with actual identifiers
        None
    }

    /// Calculate reliability from training samples
    pub fn calculate_reliability(sample_count: usize, modality: Modality) -> f32 {
        // Reliability increases with samples up to a maximum
        let base_reliability = modality.default_reliability();
        let max_samples = 100.0;
        let sample_factor = (sample_count as f32 / max_samples).min(1.0);

        // Interpolate between 50% and base reliability based on samples
        0.5 + (base_reliability - 0.5) * sample_factor
    }

    /// Check if modalities meet verification requirements
    ///
    /// Requires minimum 2 out of 3 modalities, with Face OR Voice required.
    /// This prevents verification from typing alone.
    ///
    /// Valid combinations:
    /// - Face + Voice ✓
    /// - Face + Typing ✓
    /// - Voice + Typing ✓
    /// - Typing only ✗
    pub fn meets_verification_requirements(modalities: &[ModalityConfidence]) -> bool {
        if modalities.len() < 2 {
            return false;
        }

        let has_face = modalities.iter().any(|m| m.modality == Modality::Face);
        let has_voice = modalities.iter().any(|m| m.modality == Modality::Voice);

        // Must have at least Face OR Voice (not just typing)
        has_face || has_voice
    }
}

/// Training session for a new user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSession {
    pub user_id: String,
    pub started: DateTime<Utc>,
    pub completed: Option<DateTime<Utc>>,

    // Training status for each modality
    pub typing_samples: usize,
    pub voice_samples: usize,
    pub face_samples: usize,

    // Minimum samples required
    pub min_typing_samples: usize,
    pub min_voice_samples: usize,
    pub min_face_samples: usize,

    // Social verification
    pub relationships: Vec<String>,
    pub verified_by: Option<String>, // Who verified this user
}

impl TrainingSession {
    /// Create new training session
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            started: Utc::now(),
            completed: None,
            typing_samples: 0,
            voice_samples: 0,
            face_samples: 0,
            min_typing_samples: 50, // ~500 words of typing
            min_voice_samples: 10,  // ~10 voice samples
            min_face_samples: 5,    // ~5 face captures
            relationships: Vec::new(),
            verified_by: None,
        }
    }

    /// Check if typing training is complete
    pub fn is_typing_complete(&self) -> bool {
        self.typing_samples >= self.min_typing_samples
    }

    /// Check if voice training is complete
    pub fn is_voice_complete(&self) -> bool {
        self.voice_samples >= self.min_voice_samples
    }

    /// Check if face training is complete
    pub fn is_face_complete(&self) -> bool {
        self.face_samples >= self.min_face_samples
    }

    /// Check if all training is complete
    pub fn is_complete(&self) -> bool {
        self.is_typing_complete() && self.is_voice_complete() && self.is_face_complete()
    }

    /// Get training progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        let typing_progress =
            (self.typing_samples as f32 / self.min_typing_samples as f32).min(1.0);
        let voice_progress = (self.voice_samples as f32 / self.min_voice_samples as f32).min(1.0);
        let face_progress = (self.face_samples as f32 / self.min_face_samples as f32).min(1.0);

        (typing_progress + voice_progress + face_progress) / 3.0
    }

    /// Get status message
    pub fn status(&self) -> String {
        if self.is_complete() {
            "Training complete! ✅".to_string()
        } else {
            let mut missing = Vec::new();
            if !self.is_typing_complete() {
                missing.push(format!(
                    "typing ({}/{})",
                    self.typing_samples, self.min_typing_samples
                ));
            }
            if !self.is_voice_complete() {
                missing.push(format!(
                    "voice ({}/{})",
                    self.voice_samples, self.min_voice_samples
                ));
            }
            if !self.is_face_complete() {
                missing.push(format!(
                    "face ({}/{})",
                    self.face_samples, self.min_face_samples
                ));
            }
            format!("Training: {} needed", missing.join(", "))
        }
    }

    /// Mark as complete
    pub fn complete(&mut self) {
        self.completed = Some(Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_defaults() {
        assert_eq!(Modality::Typing.name(), "Typing Pattern");
        assert_eq!(Modality::Voice.emoji(), "🎤");
        assert_eq!(Modality::Face.default_reliability(), 0.95);
    }

    #[test]
    fn test_access_level_from_confidence() {
        assert_eq!(AccessLevel::from_confidence(0.99), AccessLevel::System);
        assert_eq!(AccessLevel::from_confidence(0.95), AccessLevel::Elevated);
        assert_eq!(AccessLevel::from_confidence(0.85), AccessLevel::Standard);
        assert_eq!(AccessLevel::from_confidence(0.70), AccessLevel::Limited);
        assert_eq!(AccessLevel::from_confidence(0.50), AccessLevel::None);
    }

    #[test]
    fn test_access_level_colors() {
        assert_eq!(AccessLevel::System.color(), (255, 215, 0));
        assert_eq!(AccessLevel::Standard.color(), (255, 255, 255));
        assert_eq!(AccessLevel::None.color(), (120, 120, 120));
    }

    #[test]
    fn test_fuse_confidences() {
        let identifier = MultiModalIdentifier::default();

        let modalities = vec![
            ModalityConfidence {
                modality: Modality::Typing,
                confidence: 0.7,
                reliability: 0.7,
                sample_count: 50,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Voice,
                confidence: 0.9,
                reliability: 0.85,
                sample_count: 10,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Face,
                confidence: 0.95,
                reliability: 0.95,
                sample_count: 5,
                last_updated: Utc::now(),
            },
        ];

        let fused = identifier.fuse_confidences(&modalities);

        // Weighted average: (0.7*0.7 + 0.9*0.85 + 0.95*0.95) / (0.7 + 0.85 + 0.95)
        // = (0.49 + 0.765 + 0.9025) / 2.5 = 2.1575 / 2.5 = 0.863
        assert!(fused > 0.85 && fused < 0.87, "Fused confidence: {}", fused);
    }

    #[test]
    fn test_identify_insufficient_modalities() {
        let identifier = MultiModalIdentifier::new(2, false);

        let modalities = vec![ModalityConfidence {
            modality: Modality::Typing,
            confidence: 0.8,
            reliability: 0.7,
            sample_count: 50,
            last_updated: Utc::now(),
        }];

        let result = identifier.identify(modalities, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_identify_success() {
        let identifier = MultiModalIdentifier::default();

        let modalities = vec![
            ModalityConfidence {
                modality: Modality::Voice,
                confidence: 0.9,
                reliability: 0.85,
                sample_count: 10,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Face,
                confidence: 0.95,
                reliability: 0.95,
                sample_count: 5,
                last_updated: Utc::now(),
            },
        ];

        let result = identifier.identify(modalities, None).unwrap();
        assert!(result.fused_confidence > 0.9);
        assert_eq!(result.access_level, AccessLevel::Elevated);
    }

    #[test]
    fn test_social_verification_boost() {
        let identifier = MultiModalIdentifier::default();

        let modalities = vec![
            ModalityConfidence {
                modality: Modality::Typing,
                confidence: 0.75,
                reliability: 0.7,
                sample_count: 50,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Voice,
                confidence: 0.85,
                reliability: 0.85,
                sample_count: 10,
                last_updated: Utc::now(),
            },
        ];

        let social = SocialVerification {
            relationships: vec!["brother of system_root".to_string()],
            trust_boost: 0.1,
            inner_circle: true,
        };

        let result = identifier.identify(modalities, Some(social)).unwrap();

        // Base fusion would be ~0.81, with 0.1 boost = ~0.91
        assert!(result.fused_confidence > 0.90);
        assert_eq!(result.access_level, AccessLevel::Elevated);
        assert!(result.social_verification.is_some());
    }

    #[test]
    fn test_training_session() {
        let mut session = TrainingSession::new("test_user".to_string());

        assert!(!session.is_complete());
        assert_eq!(session.progress(), 0.0);

        // Add samples
        session.typing_samples = 50;
        assert!(session.is_typing_complete());
        assert!(!session.is_complete());

        session.voice_samples = 10;
        session.face_samples = 5;
        assert!(session.is_complete());
        assert_eq!(session.progress(), 1.0);

        session.complete();
        assert!(session.completed.is_some());
    }

    #[test]
    fn test_calculate_reliability() {
        // Few samples = lower reliability
        let reliability_low = MultiModalIdentifier::calculate_reliability(10, Modality::Voice);
        assert!(reliability_low < 0.7);

        // Many samples = higher reliability
        let reliability_high = MultiModalIdentifier::calculate_reliability(100, Modality::Voice);
        assert!(reliability_high >= 0.85);

        // Face has higher base reliability
        let face_reliability = MultiModalIdentifier::calculate_reliability(50, Modality::Face);
        let voice_reliability = MultiModalIdentifier::calculate_reliability(50, Modality::Voice);
        assert!(face_reliability > voice_reliability);
    }

    #[test]
    fn test_verification_requirements() {
        // Face + Voice = Valid
        let face_voice = vec![
            ModalityConfidence {
                modality: Modality::Face,
                confidence: 0.9,
                reliability: 0.95,
                sample_count: 5,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Voice,
                confidence: 0.85,
                reliability: 0.85,
                sample_count: 10,
                last_updated: Utc::now(),
            },
        ];
        assert!(MultiModalIdentifier::meets_verification_requirements(
            &face_voice
        ));

        // Face + Typing = Valid
        let face_typing = vec![
            ModalityConfidence {
                modality: Modality::Face,
                confidence: 0.9,
                reliability: 0.95,
                sample_count: 5,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Typing,
                confidence: 0.7,
                reliability: 0.7,
                sample_count: 50,
                last_updated: Utc::now(),
            },
        ];
        assert!(MultiModalIdentifier::meets_verification_requirements(
            &face_typing
        ));

        // Voice + Typing = Valid
        let voice_typing = vec![
            ModalityConfidence {
                modality: Modality::Voice,
                confidence: 0.85,
                reliability: 0.85,
                sample_count: 10,
                last_updated: Utc::now(),
            },
            ModalityConfidence {
                modality: Modality::Typing,
                confidence: 0.7,
                reliability: 0.7,
                sample_count: 50,
                last_updated: Utc::now(),
            },
        ];
        assert!(MultiModalIdentifier::meets_verification_requirements(
            &voice_typing
        ));

        // Typing only = Invalid
        let typing_only = vec![ModalityConfidence {
            modality: Modality::Typing,
            confidence: 0.9,
            reliability: 0.7,
            sample_count: 100,
            last_updated: Utc::now(),
        }];
        assert!(!MultiModalIdentifier::meets_verification_requirements(
            &typing_only
        ));

        // Single modality = Invalid
        let single = vec![ModalityConfidence {
            modality: Modality::Face,
            confidence: 0.95,
            reliability: 0.95,
            sample_count: 10,
            last_updated: Utc::now(),
        }];
        assert!(!MultiModalIdentifier::meets_verification_requirements(
            &single
        ));
    }
}
