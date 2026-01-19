"""
Simple scheduler to run a command at a specific time.
Designed to be used with `caffeinate` on macOS to prevent sleep while waiting.

Usage:
    caffeinate -i -s python3 schedule_runner.py --time "01:30" --cmd "python3 my_script.py"
"""

import time
import argparse
import subprocess
import shlex
import sys
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description="Schedule a command to run at a specific time.")
    parser.add_argument('--time', type=str, required=True, help="Target time in HH:MM format (24h).")
    parser.add_argument('--cmd', type=str, required=True, help="Command to execute.")
    parser.add_argument('--dry-run', action='store_true', help="Print wait time but do not execute.")
    
    args = parser.parse_args()
    
    # Parse target time
    try:
        target_hour, target_minute = map(int, args.time.split(':'))
    except ValueError:
        print("Error: Time must be in HH:MM format (e.g., 14:30 or 01:30).")
        sys.exit(1)
        
    now = datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    # If target is in the past, schedule for tomorrow
    if target <= now:
        target += timedelta(days=1)
        
    wait_seconds = (target - now).total_seconds()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scheduler started.")
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target time:  {target.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Waiting for:  {int(wait_seconds // 3600)}h {int((wait_seconds % 3600) // 60)}m {int(wait_seconds % 60)}s ({wait_seconds:.1f} seconds)")
    print(f"Command:      {args.cmd}")
    print("-" * 60)
    print("Do not close this terminal. System sleep should be prevented if running with `caffeinate`.")
    print("-" * 60)
    
    if args.dry_run:
        print("[Dry Run] Would wait and then execute.")
        return

    # Sleep loop to allow Ctrl+C
    try:
        time.sleep(wait_seconds)
    except KeyboardInterrupt:
        print("\n[Aborted] Scheduler cancelled by user.")
        sys.exit(0)
        
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waking up! Executing command...")
    print("=" * 60)
    
    # Execute
    try:
        # split command for subprocess if it's a simple string, but shell=True is often easier for complex commands users type
        subprocess.run(args.cmd, shell=True, check=True)
        print("\n" + "=" * 60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Command execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Command failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[Error] Failed to execute command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
