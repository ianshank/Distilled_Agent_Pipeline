# Training Data

Training datasets for distilled agents.

## Directory Structure

- **agents/** - Agent-specific training data in JSONL format
- **samples/** - Sample and test datasets

## Agent Training Data

The `agents/` directory contains training data for various specialized agents:

### Available Datasets

- **architect_agent.jsonl** / **architect_agent_real_data.jsonl**
  - System architecture and design training data

- **devops_agent.jsonl**
  - DevOps and infrastructure operations training data

- **sqe_agent.jsonl** / **sqe_agent_real_data.jsonl**
  - Software Quality Engineering (testing, QA) training data

- **swe_agent.jsonl** / **swe_agent_real_data.jsonl**
  - Software Engineering (development, coding) training data

- **product_manager_agent_real_data.jsonl**
  - Product management and planning training data

- **vp_product_agent.jsonl**
  - VP-level product strategy training data

- **tools_agent.jsonl**
  - Tool usage and automation training data

## Data Format

Training data is stored in JSONL (JSON Lines) format. Each line is a valid JSON object.

### Supported Formats

1. **Text format:**
```json
{"text": "Complete text for training"}
```

2. **Prompt-Completion format:**
```json
{"prompt": "User query or instruction", "completion": "Expected response"}
```

## Usage

### Validate Dataset

```bash
python scripts/evaluation/validate_dataset.py --data-file data/agents/architect_agent.jsonl
```

### Convert Format

The NLM framework automatically handles format conversion during training.

### Train with Dataset

```bash
cd NLM
python -m nlm.training.cli \
  --train-file ../data/agents/swe_agent.jsonl \
  --agent-name software-engineer \
  --output-dir outputs/swe-agent
```

## Adding New Data

To add new training data:

1. Create JSONL file in `agents/` directory
2. Use either `text` or `prompt`/`completion` format
3. Validate using `validate_dataset.py`
4. Reference in training configuration

## Data Guidelines

- Keep training examples focused on specific agent roles
- Use `_real_data` suffix for production/real-world examples
- Ensure data quality and consistency
- Remove any sensitive or confidential information
