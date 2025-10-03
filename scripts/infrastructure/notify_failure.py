#!/usr/bin/env python3
"""
CI/CD Failure Notification Script
=================================

Sends notifications when training steps fail.
"""

import click
import json
import os
from datetime import datetime

@click.command()
@click.option('--role', required=True, help='Agent role that failed')
@click.option('--step', required=True, help='Failed step')
@click.option('--logs', required=True, help='Path to log file')
@click.option('--webhook', help='Slack webhook URL')
def notify_failure(role, step, logs, webhook):
    """Send failure notification"""
    
    try:
        print(f"📢 Notifying failure for {role} at step: {step}")
        
        # Read log excerpt
        log_excerpt = "No logs available"
        if os.path.exists(logs):
            with open(logs, 'r') as f:
                lines = f.readlines()
                log_excerpt = ''.join(lines[-10:])  # Last 10 lines
        
        notification = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "failed_step": step,
            "log_excerpt": log_excerpt,
            "status": "failed"
        }
        
        # Save notification
        os.makedirs('training/notifications', exist_ok=True)
        notif_file = f"training/notifications/failure_{role.replace(' ', '_').lower()}.json"
        with open(notif_file, 'w') as f:
            json.dump(notification, f, indent=2)
        
        print(f"✅ Failure notification saved to: {notif_file}")
        
    except Exception as e:
        print(f"❌ Failed to send notification: {e}")

if __name__ == "__main__":
    notify_failure() 