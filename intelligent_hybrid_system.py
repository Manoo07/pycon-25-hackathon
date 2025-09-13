#!/usr/bin/env python3
"""
Intelligent Support Ticket Assignment System with LLM Integration
PyCon25 Hackathon Project - Enhanced Version with Fallback

This system uses Google Gemini LLM to intelligently assign support tickets to agents.
If LLM is unavailable, it falls back to advanced rule-based assignment.
"""

import asyncio
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Try to import Gemini, fallback to rule-based if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available, using rule-based fallback")

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


class SkillExtractor:
    """Advanced skill extraction with comprehensive IT keywords."""

    def __init__(self):
        self.skill_keywords = {
            'Networking': [
                'network', 'networking', 'vpn', 'connection', 'connectivity',
                'ping', 'dns', 'dhcp', 'switch', 'router', 'firewall',
                'ip address', 'subnet', 'vlan', 'bandwidth', 'latency',
                'network outage', 'network drive', 'network printer'
            ],
            'VPN_Troubleshooting': [
                'vpn', 'virtual private network', 'tunnel', 'authentication',
                'concentrator', 'remote access', 'vpn client', 'vpn server',
                'vpn connection', 'vpn endpoint'
            ],
            'Hardware_Diagnostics': [
                'hardware', 'diagnostic', 'boot', 'bios', 'power supply',
                'psu', 'motherboard', 'ram', 'memory', 'hard drive', 'ssd',
                'laptop', 'desktop', 'fan', 'overheating', 'thermal',
                'usb port', 'usb-c', 'charging', 'battery', 'projector'
            ],
            'Windows_Server_2022': [
                'windows server', 'server 2022', 'windows os', 'windows 10',
                'windows 11', 'registry', 'group policy', 'gpo', 'user profile'
            ],
            'Active_Directory': [
                'active directory', 'ad', 'domain', 'user account', 'group',
                'security group', 'domain controller', 'ldap', 'authentication',
                'login', 'password reset', 'account lockout', 'account de-provisioning'
            ],
            'Microsoft_365': [
                'microsoft 365', 'm365', 'office 365', 'outlook', 'teams',
                'sharepoint', 'onedrive', 'exchange', 'email', 'mailbox',
                'calendar sync', 'unauthorized access'
            ],
            'Database_SQL': [
                'database', 'sql', 'sql server', 'query', 'backup',
                'performance', 'connection timeout', 'etl', 'data warehouse',
                'disk space'
            ],
            'Cloud_AWS': [
                'aws', 'amazon web services', 'ec2', 's3', 'lambda',
                'cloudformation', 'iam', 'vpc'
            ],
            'Cloud_Azure': [
                'azure', 'app service', 'azure portal', 'resource group',
                'azure ad', 'subscription', 'service unavailable'
            ],
            'Network_Security': [
                'security', 'firewall', 'intrusion', 'malware', 'virus',
                'antivirus', 'phishing', 'breach', 'vulnerability',
                'encryption', 'ssl', 'certificate', 'suspicious login',
                'brute-force', 'endpoint security', 'compliance'
            ],
            'Linux_Administration': [
                'linux', 'ubuntu', 'centos', 'shell', 'bash', 'permissions',
                'chmod', 'sudo', 'systemctl', 'cron', 'samba'
            ],
            'Virtualization_VMware': [
                'vmware', 'virtual machine', 'vm', 'hypervisor',
                'virtualization', 'esxi', 'vcenter'
            ],
            'Printer_Troubleshooting': [
                'printer', 'printing', 'print queue', 'driver',
                'network printer', 'print job', 'toner', 'paper jam',
                'garbled text'
            ],
            'Voice_VoIP': [
                'voip', 'voice', 'phone', 'sip', 'pbx', 'call',
                'telephone', 'dial tone', 'conference', 'unregistered'
            ],
            'Endpoint_Security': [
                'endpoint', 'compliance', 'mdm', 'mobile device management',
                'security policy', 'device management', 'endpoint not compliant'
            ],
            'Mac_OS': [
                'mac', 'macos', 'macbook', 'apple', 'osx', 'imac',
                'big sur', 'samba share'
            ],
            'PowerShell_Scripting': [
                'powershell', 'script', 'automation', 'cmdlet', 'pipeline'
            ],
            'DevOps_CI_CD': [
                'devops', 'ci/cd', 'jenkins', 'pipeline', 'deployment',
                'continuous integration', 'continuous deployment'
            ],
            'SaaS_Integrations': [
                'saas', 'integration', 'api', 'sso', 'single sign-on',
                'saml', 'oauth', 'third-party', 'salesforce', 'jira',
                'tableau', 'visio'
            ],
            'SharePoint_Online': [
                'sharepoint', 'sharepoint online', 'site collection',
                'document library', 'permissions', 'collaboration'
            ],
            'Web_Server_Apache_Nginx': [
                'web server', 'apache', 'nginx', 'website', 'url rewrite',
                '502 bad gateway', '500 internal server', '404 error',
                'dns records', 'ssl certificates'
            ]
        }

    def extract_skills(self, text: str) -> List[str]:
        """Extract relevant skills from ticket text using advanced matching."""
        text_lower = text.lower()
        matched_skills = []
        skill_scores = defaultdict(int)

        for skill, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    skill_scores[skill] += 1

        # Return skills sorted by relevance score
        for skill, score in sorted(skill_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                matched_skills.append(skill)

        return matched_skills


class AdvancedPriorityScorer:
    """Advanced priority scoring with business impact analysis."""

    def __init__(self):
        self.urgency_patterns = {
            'critical': {
                'keywords': ['critical', 'business critical', 'emergency', 'breach', 'data loss', 'outage'],
                'weight': 10.0
            },
            'high': {
                'keywords': ['urgent', 'high priority', 'production', 'down', 'failed', 'security',
                             'malware', 'compromised', 'multiple users', 'company-wide', 'public-facing'],
                'weight': 8.0
            },
            'medium': {
                'keywords': ['cannot work', 'preventing', 'blocking', 'department', 'meeting',
                             'presentation', 'deadline', 'slow', 'intermittent'],
                'weight': 5.0
            },
            'low': {
                'keywords': ['request', 'setup', 'new employee', 'standard', 'routine'],
                'weight': 2.0
            }
        }

        self.impact_multipliers = {
            'multiple users': 1.5,
            'all users': 2.0,
            'department': 1.3,
            'company-wide': 2.5,
            'public-facing': 1.8,
            'revenue': 2.0,
            'security': 1.5,
            'data': 1.4
        }

    def calculate_priority(self, ticket: Ticket) -> Tuple[float, str]:
        """Calculate advanced priority score."""
        text = (ticket.title + " " + ticket.description).lower()

        max_priority = 1.0
        urgency_level = "low"

        # Check urgency patterns
        for level, config in self.urgency_patterns.items():
            for keyword in config['keywords']:
                if keyword in text:
                    if config['weight'] > max_priority:
                        max_priority = config['weight']
                        urgency_level = level

        # Apply impact multipliers
        for impact, multiplier in self.impact_multipliers.items():
            if impact in text:
                max_priority *= multiplier
                break

        # Time-based urgency boost
        current_time = datetime.now().timestamp()
        time_diff = current_time - ticket.creation_timestamp
        if time_diff < 3600:  # Less than 1 hour
            max_priority *= 1.2
        elif time_diff < 86400:  # Less than 24 hours
            max_priority *= 1.1

        return min(max_priority, 20.0), urgency_level


class GeminiLLMIntegration:
    """Integration with Google Gemini for intelligent analysis."""

    def __init__(self, api_key: str):
        """Initialize Gemini LLM integration."""
        self.api_key = api_key
        self.model = None

        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.model = None
        else:
            logger.info("Gemini LLM not available, using rule-based analysis")

    async def analyze_ticket(self, ticket: Ticket) -> Dict[str, Any]:
        """Analyze ticket using LLM or fallback to rule-based analysis."""

        if not self.model:
            return self._fallback_analysis(ticket)

        prompt = f"""
        Analyze this IT support ticket and provide a JSON response:

        Title: {ticket.title}
        Description: {ticket.description[:1000]}

        Return JSON with:
        {{
            "required_skills": ["list of IT skills needed"],
            "priority_level": "critical|high|medium|low",
            "urgency_score": 1-10,
            "complexity": "simple|moderate|complex",
            "category": "hardware|software|network|security|access",
            "reasoning": "brief explanation"
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean and parse JSON
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            analysis = json.loads(response_text)
            logger.info(f"LLM analyzed ticket {ticket.ticket_id}")
            return analysis

        except Exception as e:
            logger.warning(f"LLM analysis failed for {ticket.ticket_id}: {e}")
            return self._fallback_analysis(ticket)

    def _fallback_analysis(self, ticket: Ticket) -> Dict[str, Any]:
        """Fallback rule-based analysis."""
        text = (ticket.title + " " + ticket.description).lower()

        # Determine category
        category = "other"
        if any(word in text for word in ['network', 'vpn', 'dns', 'dhcp']):
            category = "network"
        elif any(word in text for word in ['laptop', 'desktop', 'hardware', 'fan']):
            category = "hardware"
        elif any(word in text for word in ['security', 'malware', 'phishing']):
            category = "security"
        elif any(word in text for word in ['software', 'application', 'outlook']):
            category = "software"
        elif any(word in text for word in ['account', 'login', 'password']):
            category = "access"

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
            "required_skills": ["General_Support"],
            "priority_level": priority_level,
            "urgency_score": urgency_score,
            "complexity": "moderate",
            "category": category,
            "reasoning": "Rule-based analysis"
        }

    async def select_optimal_agent(self, ticket: Ticket, agents: List[Agent],
                                   current_assignments: Dict[str, int]) -> Tuple[str, str, float]:
        """Select optimal agent using LLM or advanced rule-based logic."""

        if not self.model:
            return self._fallback_agent_selection(ticket, agents, current_assignments)

        # Prepare agent data for LLM
        agent_data = []
        for agent in agents:
            load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            agent_data.append({
                'id': agent.agent_id,
                'name': agent.name,
                'skills': agent.skills,
                'load': load,
                'experience': agent.experience_level,
                'available': agent.availability_status == "Available"
            })

        prompt = f"""
        Select the best agent for this IT ticket:

        Ticket: {ticket.title}
        Required Skills: {ticket.required_skills or ['General']}
        Priority: {ticket.urgency_level}

        Agents: {json.dumps(agent_data[:5])}  # Limit for token efficiency

        Return JSON: {{"agent_id": "agent_xxx", "confidence": 0.8, "reason": "explanation"}}
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            agent_id = result.get('agent_id')
            reason = result.get('reason', 'LLM selection')
            confidence = float(result.get('confidence', 0.8))

            # Validate agent exists
            if agent_id and any(a.agent_id == agent_id for a in agents):
                return agent_id, reason, confidence
            else:
                return self._fallback_agent_selection(ticket, agents, current_assignments)

        except Exception as e:
            logger.warning(f"LLM agent selection failed: {e}")
            return self._fallback_agent_selection(ticket, agents, current_assignments)

    def _fallback_agent_selection(self, ticket: Ticket, agents: List[Agent],
                                  current_assignments: Dict[str, int]) -> Tuple[str, str, float]:
        """Advanced rule-based agent selection."""

        best_agent = None
        best_score = -1
        best_reason = ""

        for agent in agents:
            if agent.availability_status != "Available":
                continue

            score = 0
            reasons = []

            # Skill matching
            if ticket.required_skills:
                skill_matches = 0
                skill_sum = 0
                for skill in ticket.required_skills:
                    agent_skill = agent.get_skill_score(skill)
                    if agent_skill > 0:
                        skill_matches += 1
                        skill_sum += agent_skill

                if skill_matches > 0:
                    skill_score = (skill_sum / skill_matches) * \
                        (skill_matches / len(ticket.required_skills))
                    score += skill_score
                    reasons.append(f"skill match: {skill_score:.1f}")

            # Experience factor
            experience_score = agent.experience_level / 12.0 * 3.0
            score += experience_score
            reasons.append(f"experience: {experience_score:.1f}")

            # Load balancing
            current_load = agent.current_load + \
                current_assignments.get(agent.agent_id, 0)
            load_penalty = current_load * 0.5
            score -= load_penalty
            reasons.append(f"load: -{load_penalty:.1f}")

            if score > best_score:
                best_score = score
                best_agent = agent
                best_reason = f"Selected {agent.name} ({', '.join(reasons)})"

        if best_agent:
            return best_agent.agent_id, best_reason, 0.7
        else:
            # Emergency fallback
            return agents[0].agent_id, "Emergency assignment - no optimal agent", 0.3


class IntelligentTicketAssignmentSystem:
    """Main intelligent assignment system with LLM integration."""

    def __init__(self, dataset_path: str, api_key: str = None):
        """Initialize the system."""
        self.skill_extractor = SkillExtractor()
        self.priority_scorer = AdvancedPriorityScorer()
        self.llm = GeminiLLMIntegration(api_key) if api_key else None
        self.assignment_counts = defaultdict(int)

        # Load data
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.agents = [Agent(**agent_data) for agent_data in data['agents']]
        self.tickets = [Ticket(**ticket_data)
                        for ticket_data in data['tickets']]

        logger.info(
            f"Loaded {len(self.agents)} agents and {len(self.tickets)} tickets")

    async def analyze_all_tickets(self):
        """Analyze all tickets using LLM or advanced rules."""
        logger.info("Analyzing all tickets...")

        for i, ticket in enumerate(self.tickets):
            if i % 10 == 0:
                logger.info(f"Analyzed {i}/{len(self.tickets)} tickets")

            # Extract skills using rule-based extractor
            ticket.required_skills = self.skill_extractor.extract_skills(
                ticket.title + " " + ticket.description
            )

            # Calculate priority
            ticket.priority_score, ticket.urgency_level = self.priority_scorer.calculate_priority(
                ticket)

            # LLM analysis if available
            if self.llm:
                ticket.llm_analysis = await self.llm.analyze_ticket(ticket)

                # Use LLM analysis to refine priority
                if ticket.llm_analysis:
                    llm_urgency = ticket.llm_analysis.get('urgency_score', 5)
                    # Blend rule-based and LLM scores
                    ticket.priority_score = (
                        ticket.priority_score + llm_urgency) / 2

        # Sort by priority
        self.tickets.sort(key=lambda t: t.priority_score, reverse=True)
        logger.info("Completed ticket analysis")

    async def assign_tickets(self) -> List[Assignment]:
        """Assign all tickets intelligently."""
        logger.info("Starting intelligent ticket assignment...")

        await self.analyze_all_tickets()

        assignments = []

        for i, ticket in enumerate(self.tickets):
            if i % 20 == 0:
                logger.info(f"Assigned {i}/{len(self.tickets)} tickets")

            if self.llm:
                agent_id, reasoning, confidence = await self.llm.select_optimal_agent(
                    ticket, self.agents, self.assignment_counts
                )
            else:
                agent_id, reasoning, confidence = self.llm._fallback_agent_selection(
                    ticket, self.agents, self.assignment_counts
                ) if self.llm else self._simple_assignment(ticket)

            assignment = Assignment(
                ticket_id=ticket.ticket_id,
                title=ticket.title,
                assigned_agent_id=agent_id,
                rationale=reasoning,
                confidence_score=confidence,
                llm_reasoning=json.dumps(
                    ticket.llm_analysis) if ticket.llm_analysis else "Rule-based analysis"
            )

            assignments.append(assignment)
            self.assignment_counts[agent_id] += 1

        logger.info("Completed ticket assignment")
        return assignments

    def _simple_assignment(self, ticket: Ticket) -> Tuple[str, str, float]:
        """Simple fallback assignment."""
        # Find available agent with lowest load
        available_agents = [
            a for a in self.agents if a.availability_status == "Available"]
        if not available_agents:
            available_agents = self.agents

        best_agent = min(available_agents,
                         key=lambda a: a.current_load + self.assignment_counts.get(a.agent_id, 0))

        return best_agent.agent_id, f"Assigned to {best_agent.name} (load balancing)", 0.6

    def generate_output(self, assignments: List[Assignment], output_path: str):
        """Generate output file."""
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

    def print_statistics(self, assignments: List[Assignment]):
        """Print comprehensive statistics."""
        print("\n" + "="*80)
        print("INTELLIGENT TICKET ASSIGNMENT SYSTEM - RESULTS")
        print("="*80)

        # Agent workload
        agent_loads = defaultdict(int)
        for assignment in assignments:
            agent_loads[assignment.assigned_agent_id] += 1

        print("\nAgent Workload Distribution:")
        print("-" * 50)
        for agent in self.agents:
            original = agent.current_load
            new = agent_loads[agent.agent_id]
            total = original + new
            print(
                f"{agent.name:20} | Original: {original:2d} | New: {new:2d} | Total: {total:2d}")

        # Priority distribution
        priority_dist = defaultdict(int)
        for ticket in self.tickets:
            priority_dist[ticket.urgency_level] += 1

        print("\nPriority Distribution:")
        print("-" * 25)
        for priority, count in priority_dist.items():
            print(f"{priority.capitalize():10} | {count:3d}")

        # Confidence stats
        confidences = [a.confidence_score for a in assignments]
        if confidences:
            print(f"\nAssignment Confidence:")
            print(f"  Average: {sum(confidences)/len(confidences):.2f}")
            print(f"  Range: {min(confidences):.2f} - {max(confidences):.2f}")

        print("\n" + "="*80)


async def main():
    """Main function."""
    print("Intelligent Support Ticket Assignment System")
    print("PyCon25 Hackathon Project - LLM Enhanced")
    print("=" * 50)

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key and GEMINI_AVAILABLE:
        print("Warning: No Gemini API key found, using rule-based system")

    try:
        system = IntelligentTicketAssignmentSystem('dataset.json', api_key)

        print(
            f"\nSystem initialized with {len(system.agents)} agents and {len(system.tickets)} tickets")
        print("Processing assignments...")

        assignments = await system.assign_tickets()

        # Generate outputs
        system.generate_output(assignments, 'intelligent_output_result.json')

        # Show statistics
        system.print_statistics(assignments)

        print(f"\nAssignment complete!")
        print(f"Output saved to: intelligent_output_result.json")
        print(f"Total assignments: {len(assignments)}")

        if system.llm and system.llm.model:
            print("✓ LLM-powered analysis enabled")
        else:
            print("✓ Advanced rule-based analysis used")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
