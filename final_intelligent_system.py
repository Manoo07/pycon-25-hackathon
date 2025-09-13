#!/usr/bin/env python3
"""
Intelligent Support Ticket Assignment System with Working Gemini Integration
PyCon25 Hackathon Project - Final Version

This system uses Google Gemini LLM to intelligently assign support tickets to agents.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

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


@dataclass
class Assignment:
    """Represents a ticket assignment to an agent."""
    ticket_id: str
    title: str
    assigned_agent_id: str
    rationale: str
    confidence_score: float = 0.0


class WorkingGeminiIntegration:
    """Working integration with Google Gemini using correct API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = False

        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                # Test the connection
                response = genai.generate_text(
                    prompt="Test connection", max_output_tokens=10)
                self.available = True
                logger.info("Gemini LLM successfully initialized and tested")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
                self.available = False
        else:
            logger.info("Gemini not available, using rule-based analysis")

    async def analyze_ticket_with_llm(self, ticket: Ticket) -> Dict[str, Any]:
        """Analyze ticket using Gemini's generate_text method."""

        if not self.available:
            return self._fallback_analysis(ticket)

        prompt = f"""
        Analyze this IT support ticket and provide a structured analysis:

        Title: {ticket.title}
        Description: {ticket.description[:800]}

        Based on this ticket, determine:
        1. What technical skills are required (from common IT skills)
        2. Priority level (critical/high/medium/low)
        3. Urgency score (1-10)
        4. Category (hardware/software/network/security/access)

        Respond with a structured analysis focusing on the technical requirements and business impact.
        Consider keywords like: critical, urgent, outage, security, malware, network, hardware, etc.
        """

        try:
            response = genai.generate_text(
                prompt=prompt,
                max_output_tokens=500,
                temperature=0.3
            )

            analysis_text = response.result if hasattr(
                response, 'result') else str(response)

            # Parse the response to extract structured data
            analysis = self._parse_llm_response(analysis_text, ticket)
            logger.info(f"LLM analyzed ticket {ticket.ticket_id}")
            return analysis

        except Exception as e:
            logger.warning(f"LLM analysis failed for {ticket.ticket_id}: {e}")
            return self._fallback_analysis(ticket)

    def _parse_llm_response(self, response_text: str, ticket: Ticket) -> Dict[str, Any]:
        """Parse LLM response to extract structured data."""
        text_lower = response_text.lower()

        # Extract priority level
        priority_level = "medium"
        urgency_score = 5

        if any(word in text_lower for word in ['critical', 'emergency', 'severe']):
            priority_level = "critical"
            urgency_score = 9
        elif any(word in text_lower for word in ['urgent', 'high', 'important']):
            priority_level = "high"
            urgency_score = 7
        elif any(word in text_lower for word in ['low', 'routine', 'standard']):
            priority_level = "low"
            urgency_score = 3

        # Extract category
        category = "software"
        if any(word in text_lower for word in ['network', 'connectivity', 'vpn']):
            category = "network"
        elif any(word in text_lower for word in ['hardware', 'laptop', 'desktop']):
            category = "hardware"
        elif any(word in text_lower for word in ['security', 'malware', 'phishing']):
            category = "security"
        elif any(word in text_lower for word in ['account', 'login', 'password']):
            category = "access"

        # Extract skills based on content analysis
        required_skills = self._extract_skills_from_analysis(
            response_text, ticket)

        return {
            "required_skills": required_skills,
            "priority_level": priority_level,
            "urgency_score": urgency_score,
            "category": category,
            "reasoning": f"LLM Analysis: {response_text[:200]}..."
        }

    def _extract_skills_from_analysis(self, analysis_text: str, ticket: Ticket) -> List[str]:
        """Extract required skills from LLM analysis and ticket content."""
        combined_text = (ticket.title + " " +
                         ticket.description + " " + analysis_text).lower()

        skill_keywords = {
            'Networking': ['network', 'vpn', 'dns', 'dhcp', 'connectivity'],
            'Hardware_Diagnostics': ['hardware', 'laptop', 'desktop', 'boot', 'fan'],
            'Windows_Server_2022': ['windows', 'server', 'registry'],
            'Active_Directory': ['active directory', 'domain', 'user account'],
            'Microsoft_365': ['outlook', 'teams', 'sharepoint', 'office'],
            'Database_SQL': ['database', 'sql', 'backup'],
            'Cloud_Azure': ['azure', 'app service'],
            'Network_Security': ['security', 'firewall', 'malware', 'phishing'],
            'Linux_Administration': ['linux', 'ubuntu', 'permissions'],
            'Printer_Troubleshooting': ['printer', 'printing'],
            'Voice_VoIP': ['voip', 'phone', 'call'],
            'Mac_OS': ['mac', 'macos', 'macbook']
        }

        matched_skills = []
        for skill, keywords in skill_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                matched_skills.append(skill)

        return matched_skills if matched_skills else ['General_Support']

    def _fallback_analysis(self, ticket: Ticket) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available."""
        text = (ticket.title + " " + ticket.description).lower()

        # Determine priority
        priority_level = "medium"
        urgency_score = 5

        if any(word in text for word in ['critical', 'outage', 'down', 'emergency']):
            priority_level = "critical"
            urgency_score = 9
        elif any(word in text for word in ['urgent', 'high', 'production']):
            priority_level = "high"
            urgency_score = 7
        elif any(word in text for word in ['request', 'setup', 'new']):
            priority_level = "low"
            urgency_score = 3

        return {
            "required_skills": self._extract_skills_from_analysis("", ticket),
            "priority_level": priority_level,
            "urgency_score": urgency_score,
            "category": "software",
            "reasoning": "Rule-based fallback analysis"
        }

    async def select_agent_with_llm(self, ticket: Ticket, agents: List[Agent],
                                    current_assignments: Dict[str, int]) -> Tuple[str, str, float]:
        """Select optimal agent using LLM reasoning."""

        if not self.available:
            return self._fallback_agent_selection(ticket, agents, current_assignments)

        # Prepare agent summary for LLM
        agent_summary = []
        for agent in agents[:5]:  # Limit to top 5 agents for efficiency
            current_load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            agent_summary.append(
                f"{agent.agent_id}: {agent.name}, Skills: {list(agent.skills.keys())[:3]}, "
                f"Load: {current_load}, Experience: {agent.experience_level}"
            )

        prompt = f"""
        Select the best IT support agent for this ticket:

        Ticket: {ticket.title}
        Required Skills: {ticket.required_skills or ['General']}
        Priority: {ticket.urgency_level}

        Available Agents:
        {chr(10).join(agent_summary)}

        Consider:
        1. Agent skills matching ticket requirements
        2. Current workload (lower is better)
        3. Experience level for complex issues
        4. Availability status

        Recommend the agent ID and explain why in 1-2 sentences.
        """

        try:
            response = genai.generate_text(
                prompt=prompt,
                max_output_tokens=200,
                temperature=0.2
            )

            response_text = response.result if hasattr(
                response, 'result') else str(response)

            # Extract agent ID from response
            agent_id = self._extract_agent_id(response_text, agents)
            if agent_id:
                confidence = 0.85
                reasoning = f"LLM Selection: {response_text[:150]}..."
                return agent_id, reasoning, confidence
            else:
                return self._fallback_agent_selection(ticket, agents, current_assignments)

        except Exception as e:
            logger.warning(f"LLM agent selection failed: {e}")
            return self._fallback_agent_selection(ticket, agents, current_assignments)

    def _extract_agent_id(self, response_text: str, agents: List[Agent]) -> str:
        """Extract agent ID from LLM response."""
        response_lower = response_text.lower()

        for agent in agents:
            if agent.agent_id.lower() in response_lower or agent.name.lower() in response_lower:
                return agent.agent_id

        return None

    def _fallback_agent_selection(self, ticket: Ticket, agents: List[Agent],
                                  current_assignments: Dict[str, int]) -> Tuple[str, str, float]:
        """Smart fallback agent selection."""

        available_agents = [
            a for a in agents if a.availability_status == "Available"]
        if not available_agents:
            available_agents = agents

        best_agent = None
        best_score = -1

        for agent in available_agents:
            score = 0

            # Skill matching
            if ticket.required_skills:
                for skill in ticket.required_skills:
                    score += agent.get_skill_score(skill)

            # Experience bonus
            score += agent.experience_level

            # Load penalty
            current_load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            score -= current_load * 2

            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            reasoning = f"Selected {best_agent.name} based on skill match and workload balance"
            return best_agent.agent_id, reasoning, 0.7
        else:
            return agents[0].agent_id, "Default assignment", 0.5


class FinalIntelligentSystem:
    """Final intelligent ticket assignment system."""

    def __init__(self, dataset_path: str, api_key: str = None):
        self.llm = WorkingGeminiIntegration(api_key) if api_key else None
        self.assignment_counts = defaultdict(int)

        # Load data
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.agents = [Agent(**agent_data) for agent_data in data['agents']]
        self.tickets = [Ticket(**ticket_data)
                        for ticket_data in data['tickets']]

        logger.info(
            f"Loaded {len(self.agents)} agents and {len(self.tickets)} tickets")

    async def process_assignments(self) -> List[Assignment]:
        """Process all ticket assignments."""
        logger.info("Starting intelligent ticket assignment...")

        # Analyze tickets
        for i, ticket in enumerate(self.tickets):
            if i % 25 == 0:
                logger.info(f"Analyzing ticket {i+1}/{len(self.tickets)}")

            if self.llm:
                ticket.llm_analysis = await self.llm.analyze_ticket_with_llm(ticket)
                if ticket.llm_analysis:
                    ticket.required_skills = ticket.llm_analysis.get(
                        'required_skills', [])
                    ticket.urgency_level = ticket.llm_analysis.get(
                        'priority_level', 'medium')
                    ticket.priority_score = float(
                        ticket.llm_analysis.get('urgency_score', 5))

        # Sort by priority
        self.tickets.sort(key=lambda t: t.priority_score, reverse=True)

        # Assign tickets
        assignments = []
        for i, ticket in enumerate(self.tickets):
            if i % 25 == 0:
                logger.info(f"Assigning ticket {i+1}/{len(self.tickets)}")

            if self.llm:
                agent_id, reasoning, confidence = await self.llm.select_agent_with_llm(
                    ticket, self.agents, self.assignment_counts
                )
            else:
                # Simple fallback
                agent = min(self.agents,
                            key=lambda a: a.current_load + self.assignment_counts.get(a.agent_id, 0))
                agent_id = agent.agent_id
                reasoning = f"Assigned to {agent.name} (load balancing)"
                confidence = 0.6

            assignment = Assignment(
                ticket_id=ticket.ticket_id,
                title=ticket.title,
                assigned_agent_id=agent_id,
                rationale=reasoning,
                confidence_score=confidence
            )

            assignments.append(assignment)
            self.assignment_counts[agent_id] += 1

        logger.info("Completed ticket assignment")
        return assignments

    def generate_output(self, assignments: List[Assignment], output_path: str):
        """Generate final output file."""
        output_data = []

        for assignment in assignments:
            output_data.append({
                "ticket_id": assignment.ticket_id,
                "title": assignment.title,
                "assigned_agent_id": assignment.assigned_agent_id,
                "rationale": assignment.rationale
            })

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def show_results(self, assignments: List[Assignment]):
        """Display assignment results."""
        print("\n" + "="*70)
        print("FINAL INTELLIGENT TICKET ASSIGNMENT RESULTS")
        print("="*70)

        # Agent workload summary
        workload = defaultdict(int)
        for assignment in assignments:
            workload[assignment.assigned_agent_id] += 1

        print("\nFinal Agent Workload:")
        print("-" * 40)
        for agent in self.agents:
            original = agent.current_load
            new = workload[agent.agent_id]
            total = original + new
            print(f"{agent.name:18} | +{new:2d} → Total: {total:2d}")

        # System performance
        if self.llm and self.llm.available:
            print("\n✓ LLM-powered intelligent analysis")
        else:
            print("\n✓ Advanced rule-based analysis")

        avg_confidence = sum(
            a.confidence_score for a in assignments) / len(assignments)
        print(f"✓ Average confidence: {avg_confidence:.2f}")

        print("="*70)


async def main():
    """Main execution function."""
    print("🚀 Intelligent Support Ticket Assignment System")
    print("📊 PyCon25 Hackathon - Final LLM-Enhanced Version")
    print("=" * 60)

    api_key = os.getenv('GEMINI_API_KEY')

    try:
        system = FinalIntelligentSystem('dataset.json', api_key)

        print(
            f"\n📈 Processing {len(system.tickets)} tickets with {len(system.agents)} agents")

        assignments = await system.process_assignments()

        # Generate output
        system.generate_output(assignments, 'final_output_result.json')

        # Show results
        system.show_results(assignments)

        print(f"\n🎉 Assignment completed successfully!")
        print(f"📄 Results saved to: final_output_result.json")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
