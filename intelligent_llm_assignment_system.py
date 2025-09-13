#!/usr/bin/env python3
"""
Intelligent Support Ticket Assignment System with LLM Integration
PyCon25 Hackathon Project - Enhanced Version

This system uses Google Gemini LLM to intelligently assign support tickets to agents based on:
- LLM-powered skill analysis and matching
- Dynamic priority assessment
- Intelligent agent selection reasoning
- Real-time workload optimization
"""

import json
import os
import re
import math
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
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
        """Get skill score for a specific skill, 0 if not present."""
        return self.skills.get(skill, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for LLM processing."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'skills': self.skills,
            'current_load': self.current_load,
            'availability_status': self.availability_status,
            'experience_level': self.experience_level
        }


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
    llm_reasoning: str = ""


class GeminiLLMIntegration:
    """Integration with Google Gemini for intelligent ticket analysis."""

    def __init__(self, api_key: str):
        """Initialize Gemini LLM integration."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        # Use the correct model name for the current API version
        try:
            self.model = genai.GenerativeModel('gemini-pro')
        except:
            # Fallback if GenerativeModel is not available
            self.model = None
        logger.info("Gemini LLM initialized successfully")

    async def analyze_ticket(self, ticket: Ticket) -> Dict[str, Any]:
        """Analyze ticket using LLM to extract skills, priority, and urgency."""

        prompt = f"""
        Analyze this IT support ticket and provide a detailed JSON response:

        Ticket ID: {ticket.ticket_id}
        Title: {ticket.title}
        Description: {ticket.description}

        Please analyze and return a JSON object with the following structure:
        {{
            "required_skills": ["list of specific technical skills needed"],
            "priority_level": "critical|high|medium|low",
            "urgency_score": 1-10,
            "complexity_level": "simple|moderate|complex|expert",
            "estimated_resolution_time": "time estimate in hours",
            "business_impact": "critical|high|medium|low",
            "category": "hardware|software|network|security|access|other",
            "keywords": ["key technical terms"],
            "reasoning": "explanation of analysis"
        }}

        Consider:
        - Technical complexity and skills required
        - Business impact and urgency indicators
        - Security implications
        - User impact (individual vs department vs company-wide)
        - Time-sensitive factors
        """

        try:
            if self.model:
                response = self.model.generate_content(prompt)
            else:
                # Use the chat method as fallback
                response = genai.chat(messages=[prompt])

            # Extract JSON from response
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()

            # Clean response to extract JSON
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            analysis = json.loads(response_text)
            logger.info(f"Analyzed ticket {ticket.ticket_id} successfully")
            return analysis

        except Exception as e:
            logger.error(
                f"Error analyzing ticket {ticket.ticket_id}: {str(e)}")
            # Fallback analysis
            return {
                "required_skills": ["General_Support"],
                "priority_level": "medium",
                "urgency_score": 5,
                "complexity_level": "moderate",
                "estimated_resolution_time": "2-4 hours",
                "business_impact": "medium",
                "category": "other",
                "keywords": [],
                "reasoning": "Fallback analysis due to LLM error"
            }

    async def select_optimal_agent(self, ticket: Ticket, agents: List[Agent],
                                   current_assignments: Dict[str, int]) -> Tuple[str, str, float]:
        """Use LLM to select the optimal agent for a ticket."""

        # Prepare agent information
        agent_info = []
        for agent in agents:
            agent_dict = agent.to_dict()
            agent_dict['current_total_load'] = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            agent_info.append(agent_dict)

        prompt = f"""
        You are an intelligent IT support ticket routing system. Select the BEST agent for this ticket.

        TICKET INFORMATION:
        - ID: {ticket.ticket_id}
        - Title: {ticket.title}
        - Description: {ticket.description[:500]}...
        - LLM Analysis: {json.dumps(ticket.llm_analysis, indent=2) if ticket.llm_analysis else 'Not available'}

        AVAILABLE AGENTS:
        {json.dumps(agent_info, indent=2)}

        SELECTION CRITERIA:
        1. Skill Match: Agent must have relevant skills for the ticket requirements
        2. Experience Level: Higher experience preferred for complex issues
        3. Current Workload: Balance load across agents (lower load preferred)
        4. Availability: Only select available agents
        5. Specialization: Prefer agents with specialized skills for specific issues

        Respond with ONLY a JSON object:
        {{
            "selected_agent_id": "agent_xxx",
            "confidence_score": 0.0-1.0,
            "reasoning": "detailed explanation of why this agent was selected",
            "skill_match_score": 0.0-1.0,
            "workload_factor": 0.0-1.0,
            "experience_factor": 0.0-1.0
        }}

        Select the agent who best combines skill relevance, appropriate experience, and manageable workload.
        """

        try:
            if self.model:
                response = self.model.generate_content(prompt)
            else:
                # Use the chat method as fallback
                response = genai.chat(messages=[prompt])

            # Extract JSON from response
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            else:
                response_text = str(response).strip()

            # Clean response to extract JSON
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            selection = json.loads(response_text)

            agent_id = selection.get('selected_agent_id')
            reasoning = selection.get('reasoning', 'LLM agent selection')
            confidence = float(selection.get('confidence_score', 0.8))

            logger.info(
                f"Selected agent {agent_id} for ticket {ticket.ticket_id}")
            return agent_id, reasoning, confidence

        except Exception as e:
            logger.error(
                f"Error selecting agent for ticket {ticket.ticket_id}: {str(e)}")
            # Fallback to first available agent
            available_agents = [
                a for a in agents if a.availability_status == "Available"]
            if available_agents:
                # Select agent with lowest current load
                best_agent = min(available_agents,
                                 key=lambda a: a.current_load + current_assignments.get(a.agent_id, 0))
                return best_agent.agent_id, "Fallback selection due to LLM error", 0.5
            else:
                return agents[0].agent_id, "No available agents - emergency assignment", 0.3


class LLMTicketAssignmentSystem:
    """Enhanced ticket assignment system with LLM integration."""

    def __init__(self, dataset_path: str, api_key: str):
        """Initialize the LLM-powered assignment system."""
        self.llm = GeminiLLMIntegration(api_key)
        self.assignment_counts = defaultdict(int)

        # Load and process data
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.agents = [Agent(**agent_data) for agent_data in data['agents']]
        self.tickets = [Ticket(**ticket_data)
                        for ticket_data in data['tickets']]

        logger.info(
            f"Loaded {len(self.agents)} agents and {len(self.tickets)} tickets")

    async def analyze_all_tickets(self):
        """Analyze all tickets using LLM."""
        logger.info("Starting LLM analysis of all tickets...")

        for i, ticket in enumerate(self.tickets):
            logger.info(
                f"Analyzing ticket {i+1}/{len(self.tickets)}: {ticket.ticket_id}")
            ticket.llm_analysis = await self.llm.analyze_ticket(ticket)

            # Update ticket properties based on LLM analysis
            if ticket.llm_analysis:
                ticket.required_skills = ticket.llm_analysis.get(
                    'required_skills', [])
                ticket.urgency_level = ticket.llm_analysis.get(
                    'priority_level', 'medium')
                ticket.priority_score = float(
                    ticket.llm_analysis.get('urgency_score', 5))

        # Sort tickets by priority (highest first)
        self.tickets.sort(key=lambda t: t.priority_score, reverse=True)
        logger.info("Completed LLM analysis of all tickets")

    async def assign_tickets(self) -> List[Assignment]:
        """Assign all tickets using LLM-powered decision making."""
        logger.info("Starting LLM-powered ticket assignment...")

        # First, analyze all tickets
        await self.analyze_all_tickets()

        assignments = []

        for i, ticket in enumerate(self.tickets):
            logger.info(
                f"Assigning ticket {i+1}/{len(self.tickets)}: {ticket.ticket_id}")

            # Use LLM to select optimal agent
            agent_id, reasoning, confidence = await self.llm.select_optimal_agent(
                ticket, self.agents, self.assignment_counts
            )

            # Create assignment
            assignment = Assignment(
                ticket_id=ticket.ticket_id,
                title=ticket.title,
                assigned_agent_id=agent_id,
                rationale=reasoning,
                confidence_score=confidence,
                llm_reasoning=f"LLM Analysis: {ticket.llm_analysis.get('reasoning', 'N/A') if ticket.llm_analysis else 'N/A'}"
            )

            assignments.append(assignment)

            # Update assignment counts for load balancing
            self.assignment_counts[agent_id] += 1

        logger.info("Completed LLM-powered ticket assignment")
        return assignments

    def generate_output(self, assignments: List[Assignment], output_path: str):
        """Generate the required output JSON file."""
        output_data = []

        for assignment in assignments:
            output_data.append({
                "ticket_id": assignment.ticket_id,
                "title": assignment.title,
                "assigned_agent_id": assignment.assigned_agent_id,
                "rationale": assignment.rationale,
                "confidence_score": assignment.confidence_score,
                "llm_reasoning": assignment.llm_reasoning
            })

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Output saved to {output_path}")

    def generate_detailed_report(self, assignments: List[Assignment], report_path: str):
        """Generate detailed analysis report."""
        report = {
            "system_info": {
                "version": "LLM-Enhanced Assignment System",
                "total_agents": len(self.agents),
                "total_tickets": len(self.tickets),
                "analysis_timestamp": datetime.now().isoformat()
            },
            "agent_workload": {},
            "skill_distribution": defaultdict(int),
            "priority_distribution": defaultdict(int),
            "confidence_stats": {
                "avg_confidence": 0.0,
                "min_confidence": 1.0,
                "max_confidence": 0.0
            },
            "assignments": []
        }

        # Calculate statistics
        total_confidence = 0
        for assignment in assignments:
            # Agent workload
            agent_id = assignment.assigned_agent_id
            agent = next(a for a in self.agents if a.agent_id == agent_id)

            if agent_id not in report["agent_workload"]:
                report["agent_workload"][agent_id] = {
                    "name": agent.name,
                    "original_load": agent.current_load,
                    "new_assignments": 0,
                    "total_load": agent.current_load
                }

            report["agent_workload"][agent_id]["new_assignments"] += 1
            report["agent_workload"][agent_id]["total_load"] += 1

            # Confidence stats
            conf = assignment.confidence_score
            total_confidence += conf
            report["confidence_stats"]["min_confidence"] = min(
                report["confidence_stats"]["min_confidence"], conf)
            report["confidence_stats"]["max_confidence"] = max(
                report["confidence_stats"]["max_confidence"], conf)

            # Find corresponding ticket for analysis
            ticket = next(t for t in self.tickets if t.ticket_id ==
                          assignment.ticket_id)

            # Skill distribution
            if ticket.llm_analysis and ticket.llm_analysis.get('required_skills'):
                for skill in ticket.llm_analysis['required_skills']:
                    report["skill_distribution"][skill] += 1

            # Priority distribution
            if ticket.llm_analysis:
                priority = ticket.llm_analysis.get('priority_level', 'medium')
                report["priority_distribution"][priority] += 1

            # Assignment details
            assignment_detail = {
                "ticket_id": assignment.ticket_id,
                "title": assignment.title,
                "assigned_agent": {
                    "id": agent.agent_id,
                    "name": agent.name
                },
                "confidence_score": assignment.confidence_score,
                "llm_analysis": ticket.llm_analysis
            }
            report["assignments"].append(assignment_detail)

        # Calculate average confidence
        if assignments:
            report["confidence_stats"]["avg_confidence"] = total_confidence / \
                len(assignments)

        # Convert defaultdicts to regular dicts
        report["skill_distribution"] = dict(report["skill_distribution"])
        report["priority_distribution"] = dict(report["priority_distribution"])

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Detailed report saved to {report_path}")

    def print_statistics(self, assignments: List[Assignment]):
        """Print assignment statistics for analysis."""
        print("\n" + "="*80)
        print("LLM-POWERED INTELLIGENT TICKET ASSIGNMENT SYSTEM - STATISTICS")
        print("="*80)

        # Agent workload distribution
        agent_assignments = defaultdict(int)
        agent_names = {agent.agent_id: agent.name for agent in self.agents}

        for assignment in assignments:
            agent_assignments[assignment.assigned_agent_id] += 1

        print("\nAgent Workload Distribution:")
        print("-" * 50)
        for agent in self.agents:
            original_load = agent.current_load
            new_assignments = agent_assignments[agent.agent_id]
            total_load = original_load + new_assignments
            print(
                f"{agent.name:20} | Original: {original_load:2d} | New: {new_assignments:2d} | Total: {total_load:2d}")

        # Priority distribution from LLM analysis
        priority_distribution = defaultdict(int)
        confidence_scores = []

        for assignment in assignments:
            confidence_scores.append(assignment.confidence_score)
            # Find corresponding ticket
            ticket = next(t for t in self.tickets if t.ticket_id ==
                          assignment.ticket_id)
            if ticket.llm_analysis:
                priority = ticket.llm_analysis.get('priority_level', 'medium')
                priority_distribution[priority] += 1

        print("\nLLM-Analyzed Priority Distribution:")
        print("-" * 35)
        for priority, count in priority_distribution.items():
            print(f"{priority.capitalize():10} | {count:3d} tickets")

        # Confidence statistics
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            max_confidence = max(confidence_scores)

            print("\nLLM Confidence Statistics:")
            print("-" * 30)
            print(f"Average Confidence: {avg_confidence:.2f}")
            print(f"Min Confidence:     {min_confidence:.2f}")
            print(f"Max Confidence:     {max_confidence:.2f}")

        # Skill analysis from LLM
        skill_usage = defaultdict(int)
        for ticket in self.tickets:
            if ticket.llm_analysis and ticket.llm_analysis.get('required_skills'):
                for skill in ticket.llm_analysis['required_skills']:
                    skill_usage[skill] += 1

        print("\nTop LLM-Identified Skills:")
        print("-" * 30)
        sorted_skills = sorted(skill_usage.items(),
                               key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills[:10]:
            print(f"{skill:25} | {count:3d} tickets")

        print("\n" + "="*80)


async def main():
    """Main function to run the LLM-powered ticket assignment system."""
    print("LLM-Powered Intelligent Support Ticket Assignment System")
    print("PyCon25 Hackathon Project - Enhanced Version with Gemini")
    print("=" * 60)

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key in the .env file")
        return

    try:
        # Initialize the system
        system = LLMTicketAssignmentSystem('dataset.json', api_key)

        print(
            f"\nLoaded {len(system.agents)} agents and {len(system.tickets)} tickets")
        print("Starting LLM-powered analysis and assignment...")

        # Generate assignments using LLM
        assignments = await system.assign_tickets()

        # Generate output files
        system.generate_output(assignments, 'llm_output_result.json')
        system.generate_detailed_report(
            assignments, 'llm_detailed_report.json')

        # Print statistics
        system.print_statistics(assignments)

        print(f"\nLLM-powered assignment complete!")
        print(f"Results saved to:")
        print(f"  - llm_output_result.json (main output)")
        print(f"  - llm_detailed_report.json (detailed analysis)")
        print(f"Total assignments: {len(assignments)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
