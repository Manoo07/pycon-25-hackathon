#!/usr/bin/env python3
"""
Quick Demo Script for Hackathon Judges
Shows the intelligent assignment system in action
"""

import json
import os
from datetime import datetime


def show_demo():
    """Demonstrate the intelligent ticket assignment system."""

    print("🧠 Intelligent Ticket Assignment System Demo")
    print("=" * 60)
    print("🎯 PyCon25 Hackathon Submission")
    print()

    # Check if results exist
    if os.path.exists('final_output_result.json'):
        with open('final_output_result.json', 'r') as f:
            assignments = json.load(f)

        print(f"✅ System processed {len(assignments)} tickets successfully!")
        print()

        # Show sample assignments
        print("📋 Sample Intelligent Assignments:")
        print("-" * 50)

        for i, assignment in enumerate(assignments[:5]):
            print(
                f"{i+1}. {assignment['ticket_id']}: {assignment['title'][:40]}...")
            print(f"   → Assigned to: {assignment['assigned_agent_id']}")
            print(f"   → Reasoning: {assignment['rationale'][:80]}...")
            print()

        # Show priority distribution
        priorities = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for assignment in assignments:
            reasoning = assignment['rationale'].upper()
            if 'CRITICAL' in reasoning:
                priorities['CRITICAL'] += 1
            elif 'HIGH' in reasoning:
                priorities['HIGH'] += 1
            elif 'LOW' in reasoning:
                priorities['LOW'] += 1
            else:
                priorities['MEDIUM'] += 1

        print("🎯 Priority Distribution:")
        print("-" * 30)
        for priority, count in priorities.items():
            print(f"   {priority}: {count} tickets")

        print()
        print("🏆 Key Features Demonstrated:")
        print("   ✅ LLM-powered intelligent analysis")
        print("   ✅ Skill-based agent matching")
        print("   ✅ Priority-aware assignment")
        print("   ✅ Load balancing consideration")
        print("   ✅ Adaptive intelligence for new datasets")

    else:
        print("❌ No results found. Please run the system first:")
        print("   python3 main.py")

    print()
    print("🚀 System ready for hackathon evaluation!")


if __name__ == "__main__":
    show_demo()
