#!/usr/bin/env python3
"""
Advanced LLM Prompt System for Intelligent Ticket Assignment
PyCon25 Hackathon - Few-Shot Learning Approach

This system uses sophisticated prompt engineering with few-shot examples
to teach the LLM how to analyze tickets and assign them to optimal agents.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Represents a support agent with skills and availability."""
    agent_id: str
    name: str
    skills: Dict[str, int]
    current_load: int
    availability_status: str
    experience_level: int

    def get_skill_score(self, skill: str) -> int:
        return self.skills.get(skill, 0)

    def to_prompt_string(self) -> str:
        """Convert agent to a concise string for LLM prompts."""
        top_skills = sorted(self.skills.items(),
                            key=lambda x: x[1], reverse=True)[:3]
        skills_str = ", ".join(
            [f"{skill}({score})" for skill, score in top_skills])
        return f"{self.agent_id}|{self.name}|Skills: {skills_str}|Load: {self.current_load}|Exp: {self.experience_level}|Status: {self.availability_status}"


@dataclass
class Ticket:
    """Represents a support ticket with content and metadata."""
    ticket_id: str
    title: str
    description: str
    creation_timestamp: int
    priority_score: float = 0.0
    required_skills: List[str] = None
    urgency_level: str = "medium"
    llm_analysis: Dict[str, Any] = None


class AdvancedLLMPromptSystem:
    """Advanced prompt engineering system for intelligent ticket assignment."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = False

        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                # Test the connection
                response = genai.generate_text(
                    prompt="Test", max_output_tokens=5)
                self.available = True
                logger.info(
                    "✅ Advanced LLM Prompt System initialized successfully")
            except Exception as e:
                logger.warning(f"❌ LLM initialization failed: {e}")
                self.available = False
        else:
            logger.info("⚠️ LLM not available, using intelligent fallback")

    def create_few_shot_examples(self) -> str:
        """Create few-shot learning examples for the LLM."""
        return """
EXAMPLE 1:
Ticket: "Email server down - Exchange not responding"
Description: "Exchange server is completely down. No one can send or receive emails. Error 'Cannot connect to server' appears in Outlook. This started 30 minutes ago and affects entire company."
Available Agents:
- agent_001|Sarah Chen|Skills: Networking(9), Linux(7), AWS(5)|Load: 2|Exp: 7|Status: Available
- agent_002|Alex Rodriguez|Skills: Windows_Server(9), Active_Directory(10), VMware(8)|Load: 1|Exp: 12|Status: Available
- agent_003|Michael Lee|Skills: Network_Security(9), Database_SQL(8), Firewall(9)|Load: 3|Exp: 10|Status: Available

ANALYSIS:
Priority: CRITICAL (company-wide email outage)
Skills Needed: Windows_Server, Active_Directory, Exchange
Best Agent: agent_002 (Alex Rodriguez)
Reasoning: Alex has highest Windows Server(9) and Active Directory(10) skills, lowest current load(1), and highest experience(12). Perfect match for Exchange server issues.
Confidence: 95%

EXAMPLE 2:
Ticket: "Cannot print to network printer"
Description: "User reports that their laptop cannot print to the HP LaserJet in the accounting department. Other users can print fine. Getting 'Printer offline' error."
Available Agents:
- agent_005|Lisa Wang|Skills: Printer_Troubleshooting(8), Hardware(6), Windows(7)|Load: 2|Exp: 5|Status: Available
- agent_006|Tom Brown|Skills: Networking(10), VPN(9), Linux(8)|Load: 4|Exp: 8|Status: Busy
- agent_007|Emma Davis|Skills: Microsoft_365(9), SharePoint(8), PowerShell(7)|Load: 1|Exp: 6|Status: Available

ANALYSIS:
Priority: LOW (single user, non-critical)
Skills Needed: Printer_Troubleshooting, Windows
Best Agent: agent_005 (Lisa Wang)
Reasoning: Lisa has specialized Printer_Troubleshooting(8) skill and moderate load(2). Though Tom has better networking skills, he's busy. Emma lacks printer expertise.
Confidence: 85%

EXAMPLE 3:
Ticket: "Database backup failed - urgent"
Description: "Nightly backup of customer database failed with error 'Transaction log full'. Production database at risk. Need immediate attention to prevent data loss."
Available Agents:
- agent_003|Michael Lee|Skills: Network_Security(9), Database_SQL(8), Firewall(9)|Load: 2|Exp: 10|Status: Available
- agent_008|David Kim|Skills: Database_SQL(10), Backup_Systems(9), Linux(7)|Load: 3|Exp: 9|Status: Available
- agent_009|Rachel Green|Skills: Cloud_Azure(8), DevOps(7), Monitoring(6)|Load: 1|Exp: 4|Status: Available

ANALYSIS:
Priority: HIGH (data integrity risk)
Skills Needed: Database_SQL, Backup_Systems
Best Agent: agent_008 (David Kim)
Reasoning: David has highest Database_SQL(10) and Backup_Systems(9) skills specifically for this issue. Though he has higher load(3), his database expertise outweighs load concerns for critical data issue.
Confidence: 92%
"""

    def create_system_prompt(self) -> str:
        """Create the system prompt with instructions and examples."""
        few_shot_examples = self.create_few_shot_examples()

        return f"""You are an expert IT Support Ticket Assignment AI. Your job is to analyze support tickets and assign them to the most suitable available agent.

INSTRUCTIONS:
1. Analyze the ticket title and description to understand the technical problem
2. Determine priority level (CRITICAL/HIGH/MEDIUM/LOW) based on business impact
3. Identify required technical skills from the problem description
4. Match agents based on: skill relevance > availability status > current workload > experience level
5. Provide clear reasoning for your choice
6. Give a confidence score (0-100%)

PRIORITY GUIDELINES:
- CRITICAL: System outages, security breaches, data loss risk, company-wide issues
- HIGH: Production problems affecting multiple users, urgent business needs
- MEDIUM: Standard issues affecting individual users or small groups
- LOW: Routine requests, non-urgent setup tasks

SKILL MATCHING PRIORITIES:
1. Direct skill match (e.g., Database_SQL for database issues)
2. Related skills (e.g., Windows_Server for Exchange issues)
3. Experience level for complex problems
4. Availability status (Available > Busy > Away)
5. Current workload (lower is better)

{few_shot_examples}

Now analyze the following ticket using the same format:
"""

    async def analyze_and_assign_with_llm(self, ticket: Ticket, agents: List[Agent],
                                          current_assignments: Dict[str, int]) -> Tuple[str, str, float, Dict[str, Any]]:
        """Use LLM with few-shot prompting to analyze and assign tickets."""

        if not self.available:
            return self._intelligent_fallback(ticket, agents, current_assignments)

        # Prepare agent data with current loads
        available_agents = []
        for agent in agents:
            current_load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            agent_copy = Agent(
                agent_id=agent.agent_id,
                name=agent.name,
                skills=agent.skills,
                current_load=current_load,
                availability_status=agent.availability_status,
                experience_level=agent.experience_level
            )
            available_agents.append(agent_copy)

        # Sort by availability and load for better selection
        available_agents.sort(key=lambda a: (
            a.availability_status != "Available", a.current_load))

        # Create agent list for prompt (limit to top 8 for efficiency)
        agent_strings = []
        for agent in available_agents[:8]:
            agent_strings.append(f"- {agent.to_prompt_string()}")

        agents_text = "\n".join(agent_strings)

        # Create the complete prompt
        system_prompt = self.create_system_prompt()

        user_prompt = f"""
Ticket: "{ticket.title}"
Description: "{ticket.description[:600]}..."
Available Agents:
{agents_text}

ANALYSIS:
"""

        full_prompt = system_prompt + user_prompt

        try:
            response = genai.generate_text(
                prompt=full_prompt,
                max_output_tokens=800,
                temperature=0.1,  # Low temperature for consistent analysis
                candidate_count=1
            )

            response_text = response.result if hasattr(
                response, 'result') else str(response)

            # Parse the LLM response
            agent_id, reasoning, confidence, analysis = self._parse_llm_assignment(
                response_text, available_agents, ticket
            )

            if agent_id:
                logger.info(
                    f"🧠 LLM assigned {ticket.ticket_id} to {agent_id} (confidence: {confidence}%)")
                return agent_id, reasoning, confidence/100, analysis
            else:
                logger.warning(
                    f"⚠️ LLM failed to select agent for {ticket.ticket_id}, using fallback")
                return self._intelligent_fallback(ticket, agents, current_assignments)

        except Exception as e:
            logger.warning(
                f"❌ LLM analysis failed for {ticket.ticket_id}: {e}")
            return self._intelligent_fallback(ticket, agents, current_assignments)

    def _parse_llm_assignment(self, response_text: str, agents: List[Agent],
                              ticket: Ticket) -> Tuple[Optional[str], str, float, Dict[str, Any]]:
        """Parse LLM response to extract assignment details."""

        # Extract priority
        priority = "MEDIUM"
        if "Priority: CRITICAL" in response_text:
            priority = "CRITICAL"
        elif "Priority: HIGH" in response_text:
            priority = "HIGH"
        elif "Priority: LOW" in response_text:
            priority = "LOW"

        # Extract confidence score
        confidence = 75.0  # default
        import re
        confidence_match = re.search(r'Confidence:\s*(\d+)%', response_text)
        if confidence_match:
            confidence = float(confidence_match.group(1))

        # Extract agent ID
        agent_id = None
        reasoning = "LLM Analysis: " + response_text[:200] + "..."

        # Look for "Best Agent: agent_XXX" pattern
        agent_match = re.search(r'Best Agent:\s*(agent_\d+)', response_text)
        if agent_match:
            potential_id = agent_match.group(1)
            # Verify this agent exists in our list
            if any(agent.agent_id == potential_id for agent in agents):
                agent_id = potential_id

        # Fallback: look for any agent_XXX mentioned
        if not agent_id:
            for agent in agents:
                if agent.agent_id in response_text:
                    agent_id = agent.agent_id
                    break

        # Extract skills if mentioned
        skills_mentioned = []
        skill_keywords = ['Database_SQL', 'Windows_Server', 'Active_Directory', 'Networking',
                          'Printer_Troubleshooting', 'Network_Security', 'Microsoft_365']
        for skill in skill_keywords:
            if skill.lower() in response_text.lower():
                skills_mentioned.append(skill)

        analysis = {
            "priority": priority,
            "confidence": confidence,
            "skills_identified": skills_mentioned,
            "llm_reasoning": response_text
        }

        return agent_id, reasoning, confidence, analysis

    def _intelligent_fallback(self, ticket: Ticket, agents: List[Agent],
                              current_assignments: Dict[str, int]) -> Tuple[str, str, float, Dict[str, Any]]:
        """Intelligent fallback when LLM is not available."""

        text = (ticket.title + " " + ticket.description).lower()

        # Determine priority and urgency
        priority_score = 5
        priority_level = "MEDIUM"

        critical_keywords = ['down', 'outage',
                             'critical', 'emergency', 'breach', 'security']
        high_keywords = ['urgent', 'high', 'production',
                         'multiple users', 'backup failed']
        low_keywords = ['request', 'setup', 'new account', 'routine']

        if any(word in text for word in critical_keywords):
            priority_score = 9
            priority_level = "CRITICAL"
        elif any(word in text for word in high_keywords):
            priority_score = 7
            priority_level = "HIGH"
        elif any(word in text for word in low_keywords):
            priority_score = 3
            priority_level = "LOW"

        # Skill-based matching
        skill_map = {
            'database': ['Database_SQL', 'Backup_Systems'],
            'email': ['Microsoft_365', 'Windows_Server', 'Active_Directory'],
            'network': ['Networking', 'VPN_Troubleshooting', 'Firewall_Configuration'],
            'printer': ['Printer_Troubleshooting', 'Hardware_Diagnostics'],
            'security': ['Network_Security', 'Firewall_Configuration', 'Identity_Management'],
            'windows': ['Windows_Server_2022', 'Active_Directory', 'PowerShell_Scripting'],
            'cloud': ['Cloud_AWS', 'Cloud_Azure', 'SaaS_Integrations'],
            'mac': ['Mac_OS', 'Hardware_Diagnostics']
        }

        required_skills = []
        for keyword, skills in skill_map.items():
            if keyword in text:
                required_skills.extend(skills)

        # Score agents
        available_agents = [
            a for a in agents if a.availability_status == "Available"]
        if not available_agents:
            available_agents = agents

        best_agent = None
        best_score = -1

        for agent in available_agents:
            score = 0

            # Skill matching (primary factor)
            for skill in required_skills:
                score += agent.get_skill_score(skill) * 2

            # Experience bonus for complex issues
            if priority_level in ["CRITICAL", "HIGH"]:
                score += agent.experience_level

            # Load penalty
            current_load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            score -= current_load * 3

            # Availability bonus
            if agent.availability_status == "Available":
                score += 10

            if score > best_score:
                best_score = score
                best_agent = agent

        if not best_agent:
            best_agent = agents[0]

        reasoning = f"Fallback Analysis: {priority_level} priority ticket requiring {required_skills[:2] if required_skills else ['general']} skills. Selected {best_agent.name} based on skill match and availability."

        analysis = {
            "priority": priority_level,
            "confidence": 65.0,
            "skills_identified": required_skills,
            "fallback_reasoning": reasoning
        }

        return best_agent.agent_id, reasoning, 0.65, analysis


class IntelligentTicketAssignmentSystem:
    """Main system orchestrating intelligent ticket assignments."""

    def __init__(self, dataset_path: str, api_key: str = None):
        self.llm_system = AdvancedLLMPromptSystem(api_key) if api_key else None
        self.assignment_counts = defaultdict(int)

        # Load data
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.agents = [Agent(**agent_data) for agent_data in data['agents']]
        self.tickets = [Ticket(**ticket_data)
                        for ticket_data in data['tickets']]

        logger.info(
            f"🎯 Loaded {len(self.agents)} agents and {len(self.tickets)} tickets")

    async def process_all_assignments(self) -> List[Dict[str, Any]]:
        """Process all ticket assignments using advanced LLM prompting."""
        logger.info("🚀 Starting advanced LLM-powered ticket assignment...")

        assignments = []
        total_confidence = 0
        llm_successes = 0

        for i, ticket in enumerate(self.tickets):
            if i % 20 == 0:
                logger.info(f"📋 Processing ticket {i+1}/{len(self.tickets)}")

            if self.llm_system:
                agent_id, reasoning, confidence, analysis = await self.llm_system.analyze_and_assign_with_llm(
                    ticket, self.agents, self.assignment_counts
                )

                if "LLM" in reasoning:
                    llm_successes += 1
            else:
                # Use fallback system
                agent_id, reasoning, confidence, analysis = self.llm_system._intelligent_fallback(
                    ticket, self.agents, self.assignment_counts
                ) if self.llm_system else self._simple_fallback(ticket)

            assignment = {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "assigned_agent_id": agent_id,
                "rationale": reasoning,
                "confidence_score": confidence,
                "analysis": analysis
            }

            assignments.append(assignment)
            self.assignment_counts[agent_id] += 1
            total_confidence += confidence

        # Log performance statistics
        avg_confidence = total_confidence / len(assignments)
        llm_success_rate = (llm_successes / len(assignments)) * \
            100 if len(assignments) > 0 else 0

        logger.info(f"✅ Assignment completed!")
        logger.info(f"📊 Average confidence: {avg_confidence:.2f}")
        logger.info(f"🧠 LLM success rate: {llm_success_rate:.1f}%")

        return assignments

    def _simple_fallback(self, ticket: Ticket) -> Tuple[str, str, float, Dict[str, Any]]:
        """Simple fallback when no advanced systems available."""
        agent = min(self.agents, key=lambda a: a.current_load +
                    self.assignment_counts.get(a.agent_id, 0))
        return (
            agent.agent_id,
            f"Simple load balancing assignment to {agent.name}",
            0.5,
            {"priority": "MEDIUM", "confidence": 50.0,
                "skills_identified": [], "fallback": True}
        )

    def generate_output(self, assignments: List[Dict[str, Any]], output_path: str):
        """Generate output file with assignments."""
        output_data = []

        for assignment in assignments:
            output_data.append({
                "ticket_id": assignment["ticket_id"],
                "title": assignment["title"],
                "assigned_agent_id": assignment["assigned_agent_id"],
                "rationale": assignment["rationale"]
            })

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"📁 Results saved to {output_path}")

    def show_detailed_results(self, assignments: List[Dict[str, Any]]):
        """Display detailed assignment results."""
        print("\n" + "="*80)
        print("🧠 ADVANCED LLM PROMPT SYSTEM - ASSIGNMENT RESULTS")
        print("="*80)

        # Performance metrics
        total_confidence = sum(a["confidence_score"] for a in assignments)
        avg_confidence = total_confidence / len(assignments)

        high_confidence = len(
            [a for a in assignments if a["confidence_score"] > 0.8])
        llm_powered = len([a for a in assignments if "LLM" in a["rationale"]])

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   • Total tickets processed: {len(assignments)}")
        print(f"   • Average confidence: {avg_confidence:.2f}")
        print(f"   • High confidence assignments (>80%): {high_confidence}")
        print(f"   • LLM-powered assignments: {llm_powered}")

        # Priority distribution
        priority_counts = defaultdict(int)
        for assignment in assignments:
            if "analysis" in assignment:
                priority = assignment["analysis"].get("priority", "UNKNOWN")
                priority_counts[priority] += 1

        print(f"\n🎯 PRIORITY DISTRIBUTION:")
        for priority, count in sorted(priority_counts.items()):
            print(f"   • {priority}: {count} tickets")

        # Agent workload
        workload = defaultdict(int)
        for assignment in assignments:
            workload[assignment["assigned_agent_id"]] += 1

        print(f"\n👥 FINAL AGENT WORKLOAD:")
        print("-" * 50)
        for agent in self.agents:
            original = agent.current_load
            new = workload[agent.agent_id]
            total = original + new
            print(f"{agent.name:20} | +{new:2d} → Total: {total:2d}")

        # Sample high-confidence assignments
        high_conf_assignments = [
            a for a in assignments if a["confidence_score"] > 0.85][:3]
        if high_conf_assignments:
            print(f"\n🌟 SAMPLE HIGH-CONFIDENCE ASSIGNMENTS:")
            print("-" * 50)
            for assignment in high_conf_assignments:
                print(
                    f"   • {assignment['ticket_id']}: {assignment['title'][:40]}...")
                print(
                    f"     → {assignment['assigned_agent_id']} (confidence: {assignment['confidence_score']:.0%})")

        print("="*80)


async def main():
    """Main execution function."""
    print("🧠 Advanced LLM Prompt System for Intelligent Ticket Assignment")
    print("🎯 PyCon25 Hackathon - Few-Shot Learning Approach")
    print("=" * 75)

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  Warning: No GEMINI_API_KEY found in environment")
        print("   The system will use intelligent fallback methods")

    try:
        system = IntelligentTicketAssignmentSystem('dataset.json', api_key)

        print(
            f"\n🔄 Processing {len(system.tickets)} tickets with {len(system.agents)} agents")
        print("🧠 Using advanced few-shot prompting for intelligent analysis...")

        assignments = await system.process_all_assignments()

        # Generate output
        system.generate_output(assignments, 'advanced_llm_output_result.json')

        # Show detailed results
        system.show_detailed_results(assignments)

        print(f"\n🎉 Advanced LLM assignment completed successfully!")
        print(f"📄 Results saved to: advanced_llm_output_result.json")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
