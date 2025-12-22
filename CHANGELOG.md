# Changelog

## [0.1.0] - 2025-12-21

### Added
- Initial release extracted from Frame project
- **Voice Recognition**: Biometric voice pattern matching
  - Voice embedding storage (384-dim vectors)
  - Cosine similarity verification
  - Baseline refinement with moving average
  - Similarity threshold: 0.85
- **Typing Dynamics**: Behavioral keystroke analysis
  - Key hold time (dwell time)
  - Flight time (key-to-key interval)
  - Typing rhythm variance analysis
  - Error correction pattern detection
  - Common n-gram timing analysis
  - Statistical baseline per user
- **Face Recognition**: Facial biometric storage (future)
- **Multimodal Fusion**: Bayesian biometric combination
  - Multi-modal identifier
  - Confidence scoring per modality
  - User consensus algorithms
  - Access level determination
- **Trust Integration**: Bridge to multi-dimensional trust
  - Voice → Trust bridge
  - Typing → Trust bridge
  - Automatic trust score updates
- **Identity Management**: User identity tracking
  - Verification status (Unverified/Pending/Verified/Trusted)
  - User roles (Owner/Admin/User/Guest)
  - Identity extraction from patterns

### Features
- Adaptive baseline refinement
- Pattern drift detection
- Multi-modal confidence scoring
- SQLite persistence for all biometrics
- Binary serialization for voice embeddings

### Modules
- identity.rs (380 LOC) - User identity management
- typing_patterns.rs (551 LOC) - Keystroke dynamics
- voice_store.rs (289 LOC) - Voice biometrics
- face_store.rs (175 LOC) - Face biometrics (future)
- multimodal.rs (400 LOC) - Multi-modal fusion
- typing_trust_bridge.rs (63 LOC) - Typing integration
- voice_trust_bridge.rs (50 LOC) - Voice integration

### Dependencies
- frame-catalog (Database, embeddings)
- frame-sentinel (Multi-dimensional trust)
- rusqlite 0.31 (persistence)
- bincode 1.3 (binary serialization)
- regex 1.10 (pattern matching)
- anyhow 1.0 (error handling)

### Notes
- Extracted from [Frame](https://github.com/Blackfall-Labs/sam)
- Production-ready for multi-modal biometric systems
- Designed to prevent impersonation and deepfake attacks
