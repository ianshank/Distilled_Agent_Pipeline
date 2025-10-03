#!/usr/bin/env python3
"""
Quick test for Software Development Agent Training Data
Validates training data without requiring full dependencies
"""

import json
import os

def test_training_data():
    """Test the training data file"""
    print("🧠 MangoMAS Software Development Agent - Quick Test")
    print("=" * 50)
    
    data_file = "sample_training_data.jsonl"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📊 Found {len(lines)} training examples")
    
    # Quick validation
    valid_count = 0
    categories = {}
    
    for line in lines:
        try:
            data = json.loads(line.strip())
            if 'prompt' in data and 'completion' in data:
                valid_count += 1
                
                # Simple categorization
                prompt_lower = data['prompt'].lower()
                if 'api' in prompt_lower or 'endpoint' in prompt_lower:
                    categories['API'] = categories.get('API', 0) + 1
                elif 'auth' in prompt_lower or 'jwt' in prompt_lower:
                    categories['Authentication'] = categories.get('Authentication', 0) + 1
                elif 'test' in prompt_lower:
                    categories['Testing'] = categories.get('Testing', 0) + 1
                elif 'deploy' in prompt_lower or 'terraform' in prompt_lower:
                    categories['Deployment'] = categories.get('Deployment', 0) + 1
                elif 'database' in prompt_lower or 'sql' in prompt_lower:
                    categories['Database'] = categories.get('Database', 0) + 1
                else:
                    categories['Other'] = categories.get('Other', 0) + 1
                    
        except json.JSONDecodeError:
            pass
    
    print(f"✅ Valid examples: {valid_count}/{len(lines)}")
    
    print(f"\n🏷️  Category Distribution:")
    for category, count in categories.items():
        percentage = (count / valid_count) * 100 if valid_count > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Show sample prompts
    print(f"\n📝 Sample Prompts:")
    sample_count = 0
    for line in lines[:5]:  # Show first 5
        try:
            data = json.loads(line.strip())
            if 'prompt' in data:
                print(f"  {sample_count + 1}. {data['prompt']}")
                sample_count += 1
        except:
            pass
    
    print(f"\n Expected Capabilities:")
    print(f"  Code generation for various frameworks")
    print(f"  Infrastructure as Code (Terraform, Kubernetes)")
    print(f"  System architecture design")
    print(f"  Security and authentication patterns")
    print(f"  Testing and monitoring strategies")
    print(f"  Database design and optimization")
    print(f"  API design and implementation")
    
    print(f"\n Training data is ready!")
    print(f" To start training (when dependencies are installed):")
    print(f"   python train_software_development_agent.py")
    
    return valid_count > 0

if __name__ == "__main__":
    test_training_data() 