//! # SAM Identity - Multi-Modal Biometric Verification
//!
//! Behavioral and biometric identity verification using voice, typing, and face recognition.
//!
//! ## Features
//!
//! - **Voice Recognition**: Biometric voice pattern matching
//! - **Typing Dynamics**: Behavioral keystroke pattern analysis
//! - **Face Recognition**: Facial biometric verification (future)
//! - **Multimodal Fusion**: Bayesian combination of multiple biometrics
//! - **Trust Integration**: Bridge to multi-dimensional trust system
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sam_identity::{VoiceStore, TypingPatternStore, MultimodalIdentity};
//!
//! // Voice authentication
//! let voice_store = VoiceStore::new(&db)?;
//! voice_store.store_voice_sample("user123", &voice_embedding)?;
//! let similarity = voice_store.verify_voice("user123", &query_embedding)?;
//!
//! // Typing pattern analysis
//! let typing_store = TypingPatternStore::new(&db)?;
//! typing_store.store_pattern("user123", &pattern)?;
//! ```

pub mod identity;
pub mod typing_patterns;
pub mod voice_store;
pub mod face_store;
pub mod multimodal;
pub mod typing_trust_bridge;
pub mod voice_trust_bridge;

// Re-export main types
pub use identity::{UserIdentity, IdentityExtractor, IdentityStore, VerificationStatus, UserRole};
pub use typing_patterns::{TypingPattern, TypingPatternStore};
pub use voice_store::{VoiceStore, VoiceStoreError};
pub use face_store::{FaceStore, FaceStoreError};
pub use multimodal::{
    MultiModalIdentifier, MultiModalResult, ModalityConfidence, Modality,
    MultiModalError, AccessLevel, SocialVerification, TrainingSession,
};
pub use typing_trust_bridge::TypingTrustBridge;
pub use voice_trust_bridge::VoiceTrustBridge;
