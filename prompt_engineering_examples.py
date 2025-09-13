#!/usr/bin/env python3
"""
One-Shot and Few-Shot Prompt Examples for Ticket Assignment
PyCon25 Hackathon - Prompt Engineering Demonstrations

This file shows different prompting strategies for using LLM as the brain
for intelligent ticket assignment based on available agents.
"""

import json
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PromptEngineeringExamples:
    """Demonstration of different prompting strategies."""

    def __init__(self):
        self.agents_data = """
Available Agents:
- agent_001|Sarah Chen|Skills: Networking(9), Linux(7), AWS(5)|Load: 2|Exp: 7|Available
- agent_002|Alex Rodriguez|Skills: Windows_Server(9), Active_Directory(10), VMware(8)|Load: 1|Exp: 12|Available  
- agent_003|Michael Lee|Skills: Network_Security(9), Database_SQL(8), Firewall(9)|Load: 3|Exp: 10|Available
- agent_004|Jessica Williams|Skills: Microsoft_365(10), SharePoint(9), PowerShell(8)|Load: 1|Exp: 8|Available
- agent_005|David Kim|Skills: Database_SQL(10), Backup_Systems(9), Linux(7)|Load: 2|Exp: 9|Available
"""

    def create_zero_shot_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a zero-shot prompt (no examples)."""
        return f"""
You are an expert IT support ticket assignment system. Analyze the ticket and assign it to the most suitable agent.

{self.agents_data}

Ticket to Assign:
Title: {ticket_title}
Description: {ticket_description}

Select the best agent and explain your reasoning in 2-3 sentences.
Format: Agent ID | Reasoning | Confidence (0-100%)
"""

    def create_one_shot_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a one-shot prompt (1 example)."""
        return f"""
You are an expert IT support ticket assignment system. Here's how to analyze tickets:

EXAMPLE:
Ticket: "Database backup failed with transaction log error"
Description: "Nightly backup failed. Error: Transaction log full. Production database at risk."
{self.agents_data}

ANALYSIS: agent_005|David Kim has Database_SQL(10) and Backup_Systems(9) skills, perfect for this database issue. Despite moderate load(2), his expertise is critical for data protection.|Confidence: 92%

Now analyze this ticket:
Title: {ticket_title}
Description: {ticket_description}
{self.agents_data}

ANALYSIS:
"""

    def create_few_shot_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a few-shot prompt (multiple examples)."""
        return f"""
You are an expert IT support ticket assignment system. Learn from these examples:

EXAMPLE 1 - Critical System Outage:
Ticket: "Email server completely down"
Description: "Exchange server crashed. 500+ users cannot send/receive emails. Business critical."
ANALYSIS: agent_002|Alex Rodriguez has Windows_Server(9) and Active_Directory(10) skills with lowest load(1) and highest experience(12). Perfect for Exchange crisis.|Confidence: 95%

EXAMPLE 2 - Individual User Issue:
Ticket: "Cannot access SharePoint site"
Description: "Single user cannot open SharePoint documents. Gets 'Access Denied' error."
ANALYSIS: agent_004|Jessica Williams has SharePoint(9) and Microsoft_365(10) expertise with low load(1). Ideal for SharePoint permissions issue.|Confidence: 88%

EXAMPLE 3 - Security Incident:
Ticket: "Suspicious network activity detected"
Description: "Firewall logs show unusual traffic patterns. Potential security breach investigation needed."
ANALYSIS: agent_003|Michael Lee has Network_Security(9) and Firewall(9) skills with security expertise. Higher load(3) acceptable for security priority.|Confidence: 90%

Now analyze this ticket using the same approach:
Title: {ticket_title}
Description: {ticket_description}
{self.agents_data}

ANALYSIS:
"""

    def create_chain_of_thought_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a prompt that encourages step-by-step reasoning."""
        return f"""
You are an expert IT support ticket assignment system. Follow this reasoning process:

{self.agents_data}

Ticket to Assign:
Title: {ticket_title}
Description: {ticket_description}

Please analyze step by step:

Step 1 - Priority Assessment:
Determine if this is CRITICAL/HIGH/MEDIUM/LOW based on business impact and urgency.

Step 2 - Skill Requirements:
Identify the specific technical skills needed to resolve this issue.

Step 3 - Agent Evaluation:
For each relevant agent, evaluate:
- Skill match score (0-10)
- Current workload impact
- Experience level for this type of issue
- Availability status

Step 4 - Final Decision:
Select the optimal agent and provide confidence score.

Analysis:
"""

    def create_role_based_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a prompt that assigns specific roles to the LLM."""
        return f"""
You are a Senior IT Operations Manager with 15 years of experience in ticket assignment optimization.

Your responsibilities:
1. Ensure critical issues get immediate expert attention
2. Balance workload across your team fairly
3. Match tickets to agents' strongest skills
4. Maintain high customer satisfaction

Your team:
{self.agents_data}

New ticket requiring assignment:
Title: {ticket_title}
Description: {ticket_description}

As an experienced manager, who would you assign this to and why?
Consider: urgency, skill match, team member capacity, and likelihood of first-call resolution.

Assignment Decision:
"""

    def create_constraint_based_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a prompt with specific constraints and rules."""
        return f"""
IT Ticket Assignment System - Rule-Based Analysis

MANDATORY CONSTRAINTS:
1. Agents with load >5 cannot take CRITICAL tickets
2. Database issues must go to agents with Database_SQL score ≥8
3. Security incidents require Network_Security or Firewall skills ≥8
4. New employee requests (load <3) get priority for LOW priority tickets
5. Experience level ≥10 required for CRITICAL issues

SCORING FORMULA:
Score = (Skill_Match * 3) + (Experience * 1) - (Current_Load * 2) + (Availability_Bonus * 2)

{self.agents_data}

Ticket Analysis Required:
Title: {ticket_title}  
Description: {ticket_description}

Apply constraints and scoring formula to select optimal agent:
"""

    def create_comparative_prompt(self, ticket_title: str, ticket_description: str) -> str:
        """Create a prompt that compares multiple agent options."""
        return f"""
Ticket Assignment Comparison Analysis

Ticket Details:
Title: {ticket_title}
Description: {ticket_description}

{self.agents_data}

Compare the top 3 most suitable agents:

Agent Option A:
- Strengths:
- Weaknesses:  
- Suitability Score (1-10):

Agent Option B:
- Strengths:
- Weaknesses:
- Suitability Score (1-10):

Agent Option C:  
- Strengths:
- Weaknesses:
- Suitability Score (1-10):

Final Recommendation:
Based on the comparison, assign to [Agent ID] because [reasoning].
Confidence: [percentage]
"""

    def demonstrate_all_prompts(self, ticket_title: str, ticket_description: str):
        """Demonstrate all prompting strategies."""
        print("="*80)
        print(f"TICKET: {ticket_title}")
        print(f"DESCRIPTION: {ticket_description}")
        print("="*80)

        prompts = {
            "Zero-Shot": self.create_zero_shot_prompt,
            "One-Shot": self.create_one_shot_prompt,
            "Few-Shot": self.create_few_shot_prompt,
            "Chain-of-Thought": self.create_chain_of_thought_prompt,
            "Role-Based": self.create_role_based_prompt,
            "Constraint-Based": self.create_constraint_based_prompt,
            "Comparative": self.create_comparative_prompt
        }

        for strategy, prompt_func in prompts.items():
            print(f"\n🧠 {strategy.upper()} PROMPTING STRATEGY:")
            print("-" * 50)
            prompt = prompt_func(ticket_title, ticket_description)
            print(prompt)
            print("\n" + "="*80)


def test_different_scenarios():
    """Test different ticket scenarios with various prompting strategies."""

    prompt_examples = PromptEngineeringExamples()

    scenarios = [
        {
            "title": "Critical database server outage",
            "description": "Primary customer database server is down. All e-commerce transactions failing. Estimated revenue loss: $10K/hour. Error: 'Cannot connect to database cluster'. Need immediate recovery."
        },
        {
            "title": "Employee laptop won't boot",
            "description": "Marketing employee's laptop shows blue screen on startup. Not business critical but user needs laptop for presentations tomorrow. No backup laptop available."
        },
        {
            "title": "Suspected malware infection",
            "description": "Finance team computer showing pop-up ads and running slowly. Employee clicked suspicious email link yesterday. May have compromised sensitive financial data."
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}:")
        prompt_examples.demonstrate_all_prompts(
            scenario["title"], scenario["description"])

        if i < len(scenarios):
            input("\nPress Enter to continue to next scenario...")


def create_custom_domain_prompt(domain: str, ticket_title: str, ticket_description: str) -> str:
    """Create domain-specific prompts for different industries."""

    domain_templates = {
        "healthcare": {
            "context": "You are assigning tickets in a healthcare IT environment where HIPAA compliance and patient safety are paramount.",
            "priorities": "Patient care systems > HIPAA compliance > General IT > Routine requests",
            "agents": "HIPAA_Specialist, Medical_Systems_Expert, General_IT_Support"
        },
        "finance": {
            "context": "You are managing IT tickets in a financial services firm where security and regulatory compliance are critical.",
            "priorities": "Trading systems > Security incidents > Compliance > General support",
            "agents": "Security_Specialist, Trading_Systems_Expert, Compliance_Officer, General_Support"
        },
        "education": {
            "context": "You are handling IT tickets in an educational institution during peak academic periods.",
            "priorities": "Learning management systems > Student services > Faculty support > Administrative tasks",
            "agents": "LMS_Administrator, Student_Tech_Support, Faculty_Liaison, General_IT"
        }
    }

    if domain not in domain_templates:
        domain = "general"
        domain_info = {
            "context": "You are managing general IT support tickets with standard business priorities.",
            "priorities": "Critical outages > Security issues > Production problems > Routine requests",
            "agents": "Senior_Engineer, Security_Specialist, General_Support, Junior_Technician"
        }
    else:
        domain_info = domain_templates[domain]

    return f"""
DOMAIN-SPECIFIC ASSIGNMENT SYSTEM: {domain.upper()}

Context: {domain_info['context']}

Priority Framework: {domain_info['priorities']}

Specialized Roles: {domain_info['agents']}

Ticket to Assign:
Title: {ticket_title}
Description: {ticket_description}

Apply domain-specific knowledge and priorities to make the optimal assignment:
"""


if __name__ == "__main__":
    print("🧠 LLM Prompting Strategies for Ticket Assignment")
    print("🎯 Choose demonstration mode:")
    print("1. Test all prompting strategies")
    print("2. Create custom domain prompt")
    print("3. Interactive prompt builder")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        test_different_scenarios()
    elif choice == "2":
        domain = input("Enter domain (healthcare/finance/education): ").strip()
        title = input("Enter ticket title: ").strip()
        description = input("Enter ticket description: ").strip()

        prompt = create_custom_domain_prompt(domain, title, description)
        print("\n🎯 CUSTOM DOMAIN PROMPT:")
        print("="*60)
        print(prompt)
    elif choice == "3":
        print("\n🔧 Interactive Prompt Builder")
        print("This would allow you to build custom prompts step by step...")
        print("(Implementation left as exercise for further customization)")
    else:
        print("Running default demonstration...")
        prompt_examples = PromptEngineeringExamples()
        prompt_examples.demonstrate_all_prompts(
            "Network connectivity issues in building A",
            "Multiple employees in building A cannot access company network. WiFi shows connected but no internet access. Started 2 hours ago."
        )
