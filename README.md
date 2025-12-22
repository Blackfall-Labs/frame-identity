# SAM Identity - Multi-Modal Biometric Verification

**Behavioral and biometric identity verification using voice, typing, and face recognition.**

Extracted from the SAM (Societal Advisory Module) project.

## Features

- **Voice Recognition**: Biometric voice pattern matching with embedding similarity
- **Typing Dynamics**: Behavioral keystroke pattern analysis (timing, pressure, rhythm)
- **Face Recognition**: Facial biometric verification (future)
- **Multimodal Fusion**: Bayesian combination of multiple biometrics
- **Trust Integration**: Bridge to multi-dimensional trust system
- **Pattern Learning**: Adaptive baseline refinement over time

## Quick Start

```toml
[dependencies]
sam-identity = "0.1.0"
```

```rust
use sam_identity::{VoiceStore, TypingPatternStore};
use sam_vector::Database;

// Initialize stores
let db = Database::new("identity.db")?;
let voice_store = VoiceStore::new(&db)?;
let typing_store = TypingPatternStore::new(&db)?;

// Voice authentication
voice_store.store_voice_sample("user123", &voice_embedding)?;
let similarity = voice_store.verify_voice("user123", &query_embedding)?;
println!("Voice similarity: {:.2}%", similarity * 100.0);

// Typing pattern analysis
let pattern = TypingPattern::from_keystrokes(&keystroke_data);
typing_store.store_pattern("user123", &pattern)?;
let match_score = typing_store.verify_pattern("user123", &pattern)?;
```

## Modules

- **identity** (380 LOC) - User identity management
- **typing_patterns** (551 LOC) - Keystroke dynamics analysis
- **voice_store** (289 LOC) - Voice biometric storage
- **face_store** (175 LOC) - Face biometric storage (future)
- **multimodal** (400 LOC) - Multi-modal fusion
- **typing_trust_bridge** (63 LOC) - Typing → trust integration
- **voice_trust_bridge** (50 LOC) - Voice → trust integration

## Voice Recognition

- Store voice embeddings (384-dim vectors)
- Cosine similarity matching
- Baseline refinement with moving average
- Similarity threshold: 0.85 for verification

## Typing Dynamics

**Pattern Features:**
- Key hold time (dwell time)
- Flight time (key-to-key interval)
- Typing rhythm variance
- Error correction patterns
- Common n-gram timings

**Analysis:**
- Statistical baseline per user
- Deviation scoring
- Pattern drift detection

## Multimodal Fusion

Combine multiple biometric modalities:

```rust
use sam_identity::{MultiModalIdentifier, Modality, ModalityConfidence};

let identifier = MultiModalIdentifier::new();

// Add modality scores
let confidences = vec![
    ModalityConfidence {
        modality: Modality::Voice,
        user_id: "user123".to_string(),
        confidence: 0.92,
    },
    ModalityConfidence {
        modality: Modality::Typing,
        user_id: "user123".to_string(),
        confidence: 0.85,
    },
];

let result = identifier.identify(&confidences)?;
println!("Verified: {} (confidence: {:.2})", result.user_id, result.confidence);
```

## Trust Integration

Bridge biometric scores to multi-dimensional trust:

```rust
use sam_identity::{VoiceTrustBridge, TypingTrustBridge};

// Voice → Trust
let voice_bridge = VoiceTrustBridge::new(voice_store, trust_manager);
voice_bridge.update_voice_trust("user123", &voice_embedding)?;

// Typing → Trust
let typing_bridge = TypingTrustBridge::new(typing_store, trust_manager);
typing_bridge.update_typing_trust("user123", &keystroke_pattern)?;
```

## Compatibility

- **Rust Edition**: 2021
- **MSRV**: 1.70+
- **Platforms**: All

## Dependencies

- `sam-vector` - Database, embeddings
- `sam-trust` - Multi-dimensional trust
- `rusqlite` (0.31) - Persistence
- `bincode` (1.3) - Binary serialization
- `regex` (1.10) - Pattern matching

## License

MIT - See [LICENSE](LICENSE) for details.

## Author

Magnus Trent <magnus@blackfall.dev>

## Links

- **GitHub:** https://github.com/Blackfall-Labs/sam-identity
- **SAM Project:** https://github.com/Blackfall-Labs/sam
