# NLM Architecture Documentation

## System Overview

The Neural Language Model (NLM) Distillation Framework is an enterprise-grade system for training smaller, specialized language models through knowledge distillation from larger teacher models. It supports local development on macOS and production training on CUDA-enabled systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                      NLM Distillation Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Teacher    │      │ Distillation │      │   Student    │  │
│  │    Model     │─────▶│   Process    │─────▶│    Model     │  │
│  │  (Frozen)    │      │              │      │  (Trainable) │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │              ┌───────▼───────┐             │          │
│         │              │  Loss = (1-α) │             │          │
│         └─────────────▶│  × CE + α ×   │◀────────────┘          │
│                        │  KL(T||S)/T²  │                        │
│                        └───────────────┘                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration Management (`nlm/config/`)

**Purpose**: Type-safe configuration with environment variable override support.

**Architecture**:
```python
┌─────────────────────────────────────────────────────────┐
│                    TrainingConfig                        │
│  (Pydantic BaseSettings with env override)              │
├─────────────────────────────────────────────────────────┤
│ Priority: CLI args > Environment > YAML > Defaults      │
│                                                           │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│ │ LoRAConfig  │  │Distillation │  │   Device    │      │
│ │   Nested    │  │   Config    │  │ Preferences │      │
│ └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

**Key Features**:
- Pydantic validation prevents type errors at parse time
- Nested configuration (LoRA, distillation) with double-underscore env vars
- Path sanitization rejects system directories
- Redacted logging excludes secrets from logs

**Data Flow**:
```
config.yaml ──┐
              ├─▶ Pydantic Merge ──▶ Validated Config ──▶ Training
ENV vars ─────┤
CLI args ─────┘
```

### 2. Data Pipeline (`nlm/data/`)

**Purpose**: Load and tokenize training data with flexible format support.

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│              Dataset Loading Pipeline                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Input JSONL                    Tokenized Dataset             │
│  ┌────────────────┐            ┌────────────────┐            │
│  │ {"prompt": X,  │            │ input_ids: [.] │            │
│  │  "completion"} │──Convert──▶│ attention_mask │──HF──▶     │
│  │                │            │ labels: [...]  │  Dataset   │
│  └────────────────┘            └────────────────┘            │
│         │                                                      │
│         │  Auto-detection:                                    │
│         │  1. --train_file (highest priority)                │
│         │  2. SM_CHANNEL_TRAIN env var                       │
│         │  3. Error if neither                               │
│         │                                                      │
└──────────────────────────────────────────────────────────────┘
```

**Format Conversion**:
```python
# Input: prompt/completion format
{"prompt": "Question", "completion": "Answer"}

# Converted to text format
{"text": "Question\nAnswer"}

# Tokenized with labels
{"input_ids": [...], "attention_mask": [...], "labels": [...]}
```

**Error Handling**:
- Skips malformed JSON lines with logging
- Warns on empty prompt or completion
- Validates file existence before processing

### 3. Model Management (`nlm/models/`)

**Purpose**: Load teacher/student models with device selection and LoRA support.

**Device Selection**:
```
┌─────────────────────────────────────────────────────────┐
│          Adaptive Device Selection                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Preference List: [cuda, mps, cpu]                      │
│                                                           │
│  ┌──────────┐  Available?                               │
│  │  CUDA    │──────Yes────▶ Select CUDA                 │
│  └──────────┘      │                                     │
│                    No                                    │
│                    │                                     │
│  ┌──────────┐     ▼  Available?                         │
│  │   MPS    │──────Yes────▶ Select MPS                  │
│  └──────────┘      │                                     │
│                    No                                    │
│                    │                                     │
│  ┌──────────┐     ▼                                     │
│  │   CPU    │────────────▶ Select CPU (fallback)        │
│  └──────────┘                                            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**LoRA Integration**:
```
┌─────────────────────────────────────────────────────────┐
│              LoRA Target Detection                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Model Architecture                                      │
│  ┌────────────────────────────────┐                     │
│  │  Layer 1: q_proj, k_proj, ... │                     │
│  │  Layer 2: c_attn, c_proj, ... │                     │
│  │  Layer 3: gate_proj, ...      │                     │
│  └────────────────────────────────┘                     │
│              │                                            │
│              ▼                                            │
│  ┌────────────────────────────────┐                     │
│  │   Auto-detect valid targets    │                     │
│  │   - Inspect module names       │                     │
│  │   - Match known patterns       │                     │
│  │   - Validate existence         │                     │
│  └────────────────────────────────┘                     │
│              │                                            │
│         Valid targets?                                   │
│              │                                            │
│      ┌───────┴────────┐                                 │
│     Yes               No                                 │
│      │                 │                                 │
│      ▼                 ▼                                 │
│  Apply LoRA    Continue without LoRA                    │
│  (log success)  (log warning)                           │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 4. Training Engine (`nlm/training/`)

**Purpose**: Execute distillation training with custom loss computation.

**Distillation Loss Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                  Distillation Loss Computation               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Student Outputs          Teacher Outputs                    │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ Logits (S)   │         │ Logits (T)   │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         │  Ground Truth Labels   │                           │
│         │  ┌──────────────┐     │                           │
│         └─▶│   [1,5,2,..] │◀────┘                           │
│            └──────┬───────┘                                  │
│                   │                                           │
│            ┌──────▼───────────────────────────┐             │
│            │  Task Loss (Cross-Entropy)       │             │
│            │  CE(S_logits, labels)            │             │
│            └──────┬───────────────────────────┘             │
│                   │                                           │
│                   │                                           │
│            ┌──────▼───────────────────────────┐             │
│            │  Distillation Loss (KL Div)      │             │
│            │  KL(softmax(T/τ)||softmax(S/τ))  │             │
│            │  × τ²                            │             │
│            └──────┬───────────────────────────┘             │
│                   │                                           │
│            ┌──────▼───────────────────────────┐             │
│            │  Total Loss                      │             │
│            │  (1-α)×CE + α×KL×τ²             │             │
│            │                                   │             │
│            │  α=0: pure task learning         │             │
│            │  α=1: pure distillation          │             │
│            │  α=0.5: balanced (default)       │             │
│            └──────────────────────────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Training Loop**:
```
Initialize
    ├─▶ Load Teacher (frozen)
    ├─▶ Load Student (trainable)
    ├─▶ Setup LoRA (optional)
    └─▶ Load Tokenizer & Dataset

For each epoch:
    For each batch:
        ├─▶ Get Student forward pass
        ├─▶ Get Teacher forward pass (no_grad)
        ├─▶ Compute distillation loss
        ├─▶ Backward pass
        ├─▶ Optimizer step
        └─▶ Log metrics

    Evaluate (if eval_dataset)
    Save checkpoint
    
Save final model
Save training metadata
```

### 5. Inference Server (`nlm/inference/`)

**Purpose**: Serve trained models via REST API with validation.

**API Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│               Inference Server Architecture              │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Client Request                                          │
│  ┌────────────────────────────────────────┐             │
│  │ POST /invocations                      │             │
│  │ {"prompt": "...", "max_length": 256}   │             │
│  └────────┬───────────────────────────────┘             │
│           │                                              │
│           ▼                                              │
│  ┌────────────────────────────────────────┐             │
│  │ Pydantic Validation                    │             │
│  │ - Type checking                        │             │
│  │ - Range validation                     │             │
│  │ - Sanitization                         │             │
│  └────────┬───────────────────────────────┘             │
│           │                                              │
│           ▼                                              │
│  ┌────────────────────────────────────────┐             │
│  │ Model Inference                        │             │
│  │ - Tokenize input                       │             │
│  │ - Generate with model                  │             │
│  │ - Decode output                        │             │
│  └────────┬───────────────────────────────┘             │
│           │                                              │
│           ▼                                              │
│  ┌────────────────────────────────────────┐             │
│  │ Response Construction                  │             │
│  │ - Validated InferenceResponse          │             │
│  │ - JSON serialization                   │             │
│  └────────┬───────────────────────────────┘             │
│           │                                              │
│           ▼                                              │
│  ┌────────────────────────────────────────┐             │
│  │ {"responses": [...],                   │             │
│  │  "input_prompt": "...",                │             │
│  │  "generation_config": {...}}           │             │
│  └────────────────────────────────────────┘             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Endpoints**:
- `GET /ping`: Health check, returns `{"status": "healthy"}`
- `POST /invocations`: Generate text from prompt with validated request/response

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────┐
│                  Security Layers                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: Input Validation                                   │
│  ┌────────────────────────────────────────────┐             │
│  │ • Pydantic schema validation                │             │
│  │ • Type enforcement                          │             │
│  │ • Range checks                              │             │
│  │ • String sanitization (null bytes, etc.)   │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  Layer 2: Path Sanitization                                  │
│  ┌────────────────────────────────────────────┐             │
│  │ • Reject system directory writes            │             │
│  │   (/etc, /sys, /proc, /root)               │             │
│  │ • Validate relative paths                   │             │
│  │ • Restrict to output_dir                    │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  Layer 3: Secrets Management                                 │
│  ┌────────────────────────────────────────────┐             │
│  │ • No hardcoded secrets                      │             │
│  │ • Environment variables only                │             │
│  │ • Redacted logging                          │             │
│  │ • Never commit credentials                  │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
│  Layer 4: Least Privilege                                    │
│  ┌────────────────────────────────────────────┐             │
│  │ • Teacher model frozen (no grad)            │             │
│  │ • Minimal file system writes                │             │
│  │ • Read-only data access                     │             │
│  └────────────────────────────────────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

### Development (Mac)
```
┌─────────────────────────────────────────────────────┐
│              Mac Development Environment             │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Hardware: Apple Silicon / Intel                     │
│  ┌───────────────────────────────────┐              │
│  │  Acceleration: MPS or CPU         │              │
│  │  Memory: 16GB+ recommended        │              │
│  │  Storage: 10GB+ free              │              │
│  └───────────────────────────────────┘              │
│                                                       │
│  Use Cases:                                          │
│  • Smoke tests with tiny models                     │
│  • Unit/integration testing                         │
│  • Development iteration                            │
│  • Data preparation                                  │
│                                                       │
│  Typical Models:                                     │
│  • Teacher: sshleifer/tiny-gpt2                     │
│  • Student: distilgpt2                              │
│  • Training time: 5-30 minutes                      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Production (CUDA PC)
```
┌─────────────────────────────────────────────────────┐
│           CUDA Production Environment                │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Hardware: NVIDIA GPU (16GB+ VRAM)                  │
│  ┌───────────────────────────────────┐              │
│  │  GPU: RTX 4090 / A100 / V100      │              │
│  │  CUDA: 12.x + cuDNN               │              │
│  │  Memory: 32GB+ system RAM         │              │
│  │  Storage: 100GB+ free             │              │
│  └───────────────────────────────────┘              │
│                                                       │
│  Use Cases:                                          │
│  • Full Granite distillation                        │
│  • Multi-agent training pipelines                   │
│  • Production model training                        │
│  • Batch processing                                  │
│                                                       │
│  Typical Models:                                     │
│  • Teacher: granite-3.0-8b-instruct                 │
│  • Student: granite-3.0-2b-instruct                 │
│  • Training time: 4-6 hours                         │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Testing Pyramid                         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│                    ┌──────────────┐                      │
│                    │ Integration  │  E2E training        │
│                    │   Tests (2)  │  with tiny models    │
│                    └──────────────┘                      │
│                   /                \                     │
│              ┌──────────────────────┐                    │
│              │   Contract Tests     │  API validation    │
│              │       (10+)          │  Flask endpoints   │
│              └──────────────────────┘                    │
│           /                            \                 │
│    ┌────────────────────────────────────────┐           │
│    │          Unit Tests (40+)              │           │
│    │  • Config validation                   │           │
│    │  • Data loading                        │           │
│    │  • Device selection                    │           │
│    │  • LoRA detection                      │           │
│    │  • Loss computation                    │           │
│    └────────────────────────────────────────┘           │
│                                                           │
│  Test Execution:                                         │
│  • Fast tests: pytest -m "not slow" (~5 seconds)        │
│  • All tests: pytest (~60 seconds)                      │
│  • CI/CD: Fast tests on every commit                    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### End-to-End Training Flow
```
┌──────────────────────────────────────────────────────────────┐
│                    Complete Data Flow                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Configuration                                             │
│     config.yaml + ENV + CLI ──▶ TrainingConfig               │
│                                                                │
│  2. Data Loading                                              │
│     JSONL file ──▶ Convert ──▶ Tokenize ──▶ HF Dataset       │
│                                                                │
│  3. Model Setup                                               │
│     ┌─ Load Teacher (frozen) ◀── HuggingFace Hub            │
│     └─ Load Student ◀──────────── HuggingFace Hub            │
│           │                                                    │
│           └─ Apply LoRA (optional) ──▶ PEFT Model            │
│                                                                │
│  4. Training Loop                                             │
│     For each batch:                                           │
│       Input ──▶ Student ──┐                                   │
│       Input ──▶ Teacher ──┤                                   │
│                           ├──▶ Distillation Loss              │
│                           │    (CE + KL divergence)           │
│                           │                                    │
│                           └──▶ Backward ──▶ Update params     │
│                                                                │
│  5. Checkpointing                                             │
│     Every N steps ──▶ Save checkpoint                         │
│     Best model ──────▶ Keep based on eval loss               │
│                                                                │
│  6. Final Output                                              │
│     outputs/agent/final/                                      │
│       ├── config.json                                         │
│       ├── pytorch_model.bin                                   │
│       ├── tokenizer files                                     │
│       ├── adapter files (if LoRA)                            │
│       └── training_metadata.json                             │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### Memory Optimization
- **LoRA**: 0.3-1% trainable parameters vs full fine-tuning
- **FP16**: 50% memory reduction on CUDA
- **Gradient Accumulation**: Effective batch size increase without memory cost
- **Gradient Checkpointing**: Trade compute for memory (not currently enabled)

### Compute Optimization
- **Device Selection**: Automatic fallback ensures utilization
- **Batch Size Tuning**: Auto-detect optimal based on VRAM
- **Multi-GPU**: DistributedDataParallel support (via PyTorch)

### Scalability
- **Horizontal**: Multi-agent training can run in parallel
- **Vertical**: Larger models with gradient checkpointing + offloading
- **Cloud**: SageMaker integration for managed training

## Future Enhancements

### Planned Features
1. **Quantization**: INT8/INT4 inference for deployment
2. **Model Registry**: MLflow integration for version tracking
3. **CI/CD Pipeline**: GitHub Actions for automated testing
4. **Multi-Dataset**: Mix multiple agent datasets in single training
5. **Curriculum Learning**: Progressive difficulty in training data
6. **Active Learning**: Sample uncertain examples for annotation

### Research Directions
1. **Multi-Teacher Distillation**: Learn from multiple teacher models
2. **Cross-Architecture**: Distill from different model families
3. **Layer-wise Distillation**: Match intermediate representations
4. **Adaptive Temperature**: Learn temperature per sample

## Monitoring and Observability

### Logging Levels
```
DEBUG   : Step-by-step execution details
INFO    : Key milestones (model loaded, training started)
WARNING : Non-fatal issues (LoRA disabled, fallback to CPU)
ERROR   : Fatal errors requiring intervention
```

### Metrics Tracked
- **Training**: Loss (total, task, distillation), learning rate, gradient norm
- **Hardware**: GPU utilization, memory usage, temperature
- **Performance**: Samples/second, tokens/second, wall-clock time
- **Quality**: Eval loss, perplexity (if applicable)

### Integration Points
- **Weights & Biases**: Automatic experiment tracking
- **TensorBoard**: Local visualization of metrics
- **Logs**: Structured JSON logs for analysis
- **Metadata**: Training config snapshot in output

## Compliance and Standards

### Code Quality
- **PEP 8**: Python style guide compliance
- **Type Hints**: Full typing coverage for IDE support
- **Docstrings**: NumPy/Google style documentation
- **SRP**: Single Responsibility Principle per module
- **DRY**: No code duplication
- **SOLID**: Clean architecture principles

### Security Standards
- **OWASP**: Input validation, output encoding
- **Least Privilege**: Minimal permissions required
- **Defense in Depth**: Multiple security layers
- **Zero Trust**: Validate all inputs

### Testing Standards
- **Coverage**: 80%+ code coverage target
- **Pyramid**: More unit tests than integration tests
- **Isolation**: Tests don't depend on external services
- **Fast**: Unit tests < 1s, integration < 60s

## Conclusion

The NLM Distillation Framework provides a production-ready solution for knowledge distillation with:
- **Flexibility**: Supports Mac and CUDA environments
- **Robustness**: Comprehensive testing and error handling
- **Security**: Multiple validation and sanitization layers
- **Scalability**: Efficient LoRA training and multi-agent support
- **Maintainability**: Clean architecture with clear separation of concerns

For implementation details, see `IMPLEMENTATION_SUMMARY.md`.  
For usage instructions, see `README.md`, `RUNBOOK_MAC.md`, and `RUNBOOK_CUDA.md`.

