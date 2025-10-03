#!/usr/bin/env python3
"""
Test script for Software Development Agent Distillation
Validates training data and demonstrates the system
"""

import json
import os
import sys
from pathlib import Path

def test_training_data(data_file: str):
    """Test and validate the training data"""
    print(f"🔍 Testing training data: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    print(f"📊 Found {len(lines)} training examples")
    
    # Analyze the data
    categories = {
        'architecture': 0,
        'authentication': 0,
        'testing': 0,
        'deployment': 0,
        'monitoring': 0,
        'database': 0,
        'api': 0,
        'security': 0,
        'other': 0
    }
    
    total_prompt_length = 0
    total_completion_length = 0
    valid_examples = 0
    
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            if 'prompt' in data and 'completion' in data:
                valid_examples += 1
                prompt = data['prompt'].lower()
                completion = data['completion']
                
                # Categorize based on prompt content
                if any(word in prompt for word in ['architecture', 'system design', 'infrastructure']):
                    categories['architecture'] += 1
                elif any(word in prompt for word in ['auth', 'login', 'jwt', 'oauth']):
                    categories['authentication'] += 1
                elif any(word in prompt for word in ['test', 'testing', 'test case']):
                    categories['testing'] += 1
                elif any(word in prompt for word in ['deploy', 'terraform', 'kubernetes', 'docker']):
                    categories['deployment'] += 1
                elif any(word in prompt for word in ['monitor', 'dashboard', 'metrics']):
                    categories['monitoring'] += 1
                elif any(word in prompt for word in ['database', 'sql', 'schema']):
                    categories['database'] += 1
                elif any(word in prompt for word in ['api', 'endpoint', 'rest']):
                    categories['api'] += 1
                elif any(word in prompt for word in ['security', 'secure', 'encrypt']):
                    categories['security'] += 1
                else:
                    categories['other'] += 1
                
                total_prompt_length += len(prompt)
                total_completion_length += len(completion)
            else:
                print(f"⚠️  Line {i}: Missing prompt or completion")
                
        except json.JSONDecodeError:
            print(f"❌ Line {i}: Invalid JSON")
    
    # Print analysis results
    print(f"\n📈 Data Analysis Results:")
    print(f"✅ Valid examples: {valid_examples}/{len(lines)}")
    print(f"📝 Average prompt length: {total_prompt_length/valid_examples:.1f} characters")
    print(f"📝 Average completion length: {total_completion_length/valid_examples:.1f} characters")
    
    print(f"\n🏷️  Category Distribution:")
    for category, count in categories.items():
        percentage = (count / valid_examples) * 100 if valid_examples > 0 else 0
        print(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
    
    return valid_examples > 0

def test_sample_predictions():
    """Test sample predictions with the training data"""
    print(f"\n🧪 Testing Sample Predictions")
    
    # Sample prompts from the training data
    sample_prompts = [
        "Write FastAPI code for a /ping endpoint.",
        "Create a CI/CD pipeline for a Python microservice.",
        "Implement rate limiting for a REST API.",
        "Design a scalable database schema for user management.",
        "Write a Kubernetes deployment manifest for a Node.js app."
    ]
    
    print(f"📝 Sample prompts that the agent will learn to handle:")
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"  {i}. {prompt}")
    
    print(f"\n💡 Expected capabilities after training:")
    print(f"  ✅ Code generation for various frameworks")
    print(f"  ✅ Infrastructure as Code (Terraform, Kubernetes)")
    print(f"  ✅ System architecture design")
    print(f"  ✅ Security and authentication patterns")
    print(f"  ✅ Testing and monitoring strategies")
    print(f"  ✅ Database design and optimization")
    print(f"  ✅ API design and implementation")

def test_training_configuration():
    """Test the training configuration"""
    print(f"\n⚙️  Training Configuration Test")
    
    # Import the configuration function
    try:
        import sys
        sys.path.append('training')
        from train_software_development_agent import create_software_dev_config
        config = create_software_dev_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"📊 Key settings:")
        print(f"  Teacher Model: {config['teacher_model_name']}")
        print(f"  Student Model: {config['student_model_name']}")
        print(f"  Epochs: {config['num_train_epochs']}")
        print(f"  Batch Size: {config['per_device_train_batch_size']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Distillation Alpha: {config['distillation_alpha']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  LoRA Enabled: {config['use_lora']}")
        print(f"  FP16 Enabled: {config['use_fp16']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import training configuration: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 MangoMAS Software Development Agent - Training Data Test")
    print("=" * 60)
    
    # Test training data
    data_file = "training/sample_training_data.jsonl"
    data_valid = test_training_data(data_file)
    
    # Test sample predictions
    test_sample_predictions()
    
    # Test training configuration
    config_valid = test_training_configuration()
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"  Training Data: {'✅ Valid' if data_valid else '❌ Invalid'}")
    print(f"  Configuration: {'✅ Valid' if config_valid else '❌ Invalid'}")
    
    if data_valid and config_valid:
        print(f"\n🎉 All tests passed! Ready for training.")
        print(f"\n🚀 To start training, run:")
        print(f"   python training/train_software_dev_agent.py")
        print(f"\n📚 For more options:")
        print(f"   python training/train_software_dev_agent.py --help")
    else:
        print(f"\n❌ Some tests failed. Please fix issues before training.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 