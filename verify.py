#!/usr/bin/env python3
"""
Verification script to confirm main.py processes dataset.json correctly
"""

import json
import os


def verify_system():
    """Verify that the system processes dataset.json correctly."""

    print("🔍 System Verification")
    print("=" * 40)

    # Check if dataset.json exists
    if not os.path.exists('dataset.json'):
        print("❌ dataset.json not found!")
        return False

    # Load and check dataset
    with open('dataset.json', 'r') as f:
        dataset = json.load(f)

    agents_count = len(dataset.get('agents', []))
    tickets_count = len(dataset.get('tickets', []))

    print(f"✅ dataset.json found")
    print(f"   📊 Agents: {agents_count}")
    print(f"   🎫 Tickets: {tickets_count}")

    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("❌ main.py not found!")
        return False

    print("✅ main.py found")

    # Check if output exists (after running main.py)
    if os.path.exists('final_output_result.json'):
        with open('final_output_result.json', 'r') as f:
            results = json.load(f)

        assignments_count = len(results)
        print(f"✅ final_output_result.json found")
        print(f"   📋 Assignments: {assignments_count}")

        if assignments_count == tickets_count:
            print("✅ All tickets have been assigned!")
        else:
            print(
                f"⚠️  Assignment mismatch: {tickets_count} tickets, {assignments_count} assignments")

    else:
        print("⚠️  final_output_result.json not found. Run 'python3 main.py' first.")

    print()
    print("🎯 To run the system:")
    print("   python3 main.py")
    print()
    print("🎭 To see demo:")
    print("   python3 demo.py")

    return True


if __name__ == "__main__":
    verify_system()
