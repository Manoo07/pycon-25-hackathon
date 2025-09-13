#!/usr/bin/env python3
"""
Intelligent Support Ticket Assignment System
PyCon25 Hackathon Project

This system optimally assigns support tickets to agents based on:
- Skill matching and expertise levels
- Workload balancing across agents
- Intelligent priority scoring
- Agent availability and experience
"""

import json
import re
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


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


@dataclass
class Assignment:
    """Represents a ticket assignment to an agent."""
    ticket_id: str
    title: str
    assigned_agent_id: str
    rationale: str
    confidence_score: float = 0.0


class SkillExtractor:
    """Extracts relevant skills from ticket content using keyword matching."""

    def __init__(self):
        # Comprehensive skill mapping for IT support tickets
        self.skill_keywords = {
            'Networking': [
                'network', 'networking', 'vpn', 'connection', 'connectivity',
                'ping', 'dns', 'dhcp', 'switch', 'router', 'firewall',
                'ip address', 'subnet', 'vlan', 'bandwidth', 'latency'
            ],
            'VPN_Troubleshooting': [
                'vpn', 'virtual private network', 'tunnel', 'authentication',
                'concentrator', 'remote access', 'vpn client', 'vpn server'
            ],
            'Hardware_Diagnostics': [
                'hardware', 'diagnostic', 'boot', 'bios', 'power supply',
                'psu', 'motherboard', 'ram', 'memory', 'hard drive', 'ssd',
                'laptop', 'desktop', 'fan', 'overheating', 'thermal'
            ],
            'Windows_Server_2022': [
                'windows server', 'server 2022', 'windows os', 'windows 10',
                'windows 11', 'registry', 'group policy', 'gpo'
            ],
            'Active_Directory': [
                'active directory', 'ad', 'domain', 'user account', 'group',
                'security group', 'domain controller', 'ldap', 'authentication',
                'login', 'password reset', 'account lockout'
            ],
            'Microsoft_365': [
                'microsoft 365', 'm365', 'office 365', 'outlook', 'teams',
                'sharepoint', 'onedrive', 'exchange', 'email', 'mailbox'
            ],
            'Database_SQL': [
                'database', 'sql', 'sql server', 'query', 'backup',
                'performance', 'connection timeout', 'etl', 'data warehouse'
            ],
            'Cloud_AWS': [
                'aws', 'amazon web services', 'ec2', 's3', 'lambda',
                'cloudformation', 'iam', 'vpc'
            ],
            'Cloud_Azure': [
                'azure', 'app service', 'azure portal', 'resource group',
                'azure ad', 'subscription'
            ],
            'Network_Security': [
                'security', 'firewall', 'intrusion', 'malware', 'virus',
                'antivirus', 'phishing', 'breach', 'vulnerability',
                'encryption', 'ssl', 'certificate'
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
                'network printer', 'print job', 'toner', 'paper jam'
            ],
            'Voice_VoIP': [
                'voip', 'voice', 'phone', 'sip', 'pbx', 'call',
                'telephone', 'dial tone', 'conference'
            ],
            'Endpoint_Security': [
                'endpoint', 'compliance', 'mdm', 'mobile device management',
                'security policy', 'device management'
            ],
            'Mac_OS': [
                'mac', 'macos', 'macbook', 'apple', 'osx', 'imac'
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
                'saml', 'oauth', 'third-party'
            ],
            'SharePoint_Online': [
                'sharepoint', 'sharepoint online', 'site collection',
                'document library', 'permissions', 'collaboration'
            ]
        }

    def extract_skills(self, text: str) -> List[str]:
        """Extract relevant skills from ticket text."""
        text_lower = text.lower()
        matched_skills = []

        for skill, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_skills.append(skill)
                    break  # Only add skill once per ticket

        return matched_skills


class PriorityScorer:
    """Calculates priority scores for tickets based on urgency indicators."""

    def __init__(self):
        # Priority keywords with weights
        self.urgency_keywords = {
            'critical': 10.0,
            'urgent': 9.0,
            'high priority': 8.5,
            'business critical': 10.0,
            'production': 8.0,
            'outage': 9.5,
            'down': 8.0,
            'failed': 7.0,
            'failure': 7.0,
            'emergency': 10.0,
            'immediate': 9.0,
            'asap': 8.0,
            'security': 8.5,
            'breach': 10.0,
            'malware': 8.0,
            'virus': 7.5,
            'phishing': 8.5,
            'compromised': 9.0,
            'locked out': 6.0,
            'cannot work': 7.0,
            'preventing': 6.5,
            'blocking': 6.5,
            'stuck': 5.0,
            'slow': 4.0,
            'intermittent': 5.5,
            'multiple users': 7.0,
            'all users': 8.5,
            'department': 6.5,
            'company-wide': 9.0,
            'public-facing': 8.0,
            'customer': 7.5,
            'revenue': 8.5,
            'data loss': 9.5,
            'corruption': 8.0,
            'meeting': 6.0,
            'presentation': 6.5,
            'deadline': 7.0,
            'new employee': 3.0,
            'setup': 2.0,
            'request': 2.0,
            'standard': 1.0,
            'routine': 1.0
        }

        # Department impact weights
        self.department_weights = {
            'finance': 1.3,
            'hr': 1.1,
            'sales': 1.2,
            'marketing': 1.1,
            'r&d': 1.2,
            'development': 1.2,
            'executive': 1.4,
            'management': 1.3
        }

    def calculate_priority(self, ticket: Ticket) -> Tuple[float, str]:
        """Calculate priority score and urgency level for a ticket."""
        text = (ticket.title + " " + ticket.description).lower()

        # Base priority from keywords
        priority_score = 1.0  # Base score
        matched_keywords = []

        for keyword, weight in self.urgency_keywords.items():
            if keyword in text:
                priority_score = max(priority_score, weight)
                matched_keywords.append(keyword)

        # Department impact multiplier
        for dept, multiplier in self.department_weights.items():
            if dept in text:
                priority_score *= multiplier
                break

        # Time-based urgency (newer tickets get slight boost)
        current_time = datetime.now().timestamp()
        time_diff = current_time - ticket.creation_timestamp
        if time_diff < 3600:  # Less than 1 hour
            priority_score *= 1.1
        elif time_diff < 86400:  # Less than 24 hours
            priority_score *= 1.05

        # Determine urgency level
        if priority_score >= 9.0:
            urgency_level = "critical"
        elif priority_score >= 7.0:
            urgency_level = "high"
        elif priority_score >= 4.0:
            urgency_level = "medium"
        else:
            urgency_level = "low"

        return min(priority_score, 15.0), urgency_level  # Cap at 15.0


class LoadBalancer:
    """Manages workload distribution across agents."""

    def __init__(self, agents: List[Agent]):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.assignment_counts = defaultdict(int)

    def calculate_load_factor(self, agent_id: str) -> float:
        """Calculate load factor for an agent (lower is better)."""
        agent = self.agents[agent_id]
        current_load = agent.current_load + self.assignment_counts[agent_id]

        # Experience-based capacity (more experienced agents can handle more)
        base_capacity = 5  # Base capacity for all agents
        experience_bonus = agent.experience_level * 0.3
        total_capacity = base_capacity + experience_bonus

        # Load factor calculation
        load_factor = current_load / total_capacity

        # Availability penalty
        if agent.availability_status != "Available":
            load_factor *= 2.0  # Penalty for unavailable agents

        return load_factor

    def assign_ticket(self, agent_id: str):
        """Record a ticket assignment to update load tracking."""
        self.assignment_counts[agent_id] += 1


class TicketAssignmentSystem:
    """Main system for intelligent ticket assignment."""

    def __init__(self, dataset_path: str):
        self.skill_extractor = SkillExtractor()
        self.priority_scorer = PriorityScorer()

        # Load and process data
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.agents = [Agent(**agent_data) for agent_data in data['agents']]
        self.load_balancer = LoadBalancer(self.agents)

        # Process tickets
        self.tickets = []
        for ticket_data in data['tickets']:
            ticket = Ticket(**ticket_data)
            # Extract skills and calculate priority
            ticket.required_skills = self.skill_extractor.extract_skills(
                ticket.title + " " + ticket.description
            )
            ticket.priority_score, ticket.urgency_level = self.priority_scorer.calculate_priority(
                ticket)
            self.tickets.append(ticket)

        # Sort tickets by priority (highest first)
        self.tickets.sort(key=lambda t: t.priority_score, reverse=True)

    def calculate_agent_score(self, agent: Agent, ticket: Ticket) -> Tuple[float, str]:
        """Calculate how well an agent matches a ticket."""

        # Skill matching score
        skill_score = 0.0
        skill_details = []

        if ticket.required_skills:
            total_skills = len(ticket.required_skills)
            matched_skills = 0
            skill_sum = 0

            for skill in ticket.required_skills:
                agent_skill_level = agent.get_skill_score(skill)
                if agent_skill_level > 0:
                    matched_skills += 1
                    skill_sum += agent_skill_level
                    skill_details.append(f"{skill}({agent_skill_level})")

            if matched_skills > 0:
                # Average skill level for matched skills
                avg_skill_level = skill_sum / matched_skills
                # Coverage ratio (how many required skills are covered)
                coverage_ratio = matched_skills / total_skills
                # Combined skill score
                skill_score = (avg_skill_level / 10.0) * coverage_ratio * 10.0
            else:
                # No direct skill match, use experience as fallback
                skill_score = agent.experience_level / 12.0 * 2.0  # Max 2.0 for experience only
        else:
            # No specific skills required, use general experience
            skill_score = agent.experience_level / 12.0 * 5.0  # Max 5.0 for general tickets

        # Load balancing factor (lower load is better)
        load_factor = self.load_balancer.calculate_load_factor(agent.agent_id)
        # Max 5.0, decreases with load
        load_score = max(0, 5.0 - load_factor * 2.0)

        # Experience bonus
        experience_score = (agent.experience_level / 12.0) * 2.0  # Max 2.0

        # Availability bonus
        availability_score = 1.0 if agent.availability_status == "Available" else 0.0

        # Total score calculation
        total_score = skill_score + load_score + experience_score + availability_score

        # Create rationale
        skill_text = f"skills in {', '.join(skill_details)}" if skill_details else "general experience"
        rationale = (
            f"Assigned to {agent.name} ({agent.agent_id}) based on {skill_text}, "
            f"experience level {agent.experience_level}, and current workload {agent.current_load + self.load_balancer.assignment_counts[agent.agent_id]}"
        )

        return total_score, rationale

    def assign_tickets(self) -> List[Assignment]:
        """Assign all tickets to optimal agents."""
        assignments = []

        for ticket in self.tickets:
            best_agent = None
            best_score = -1
            best_rationale = ""

            # Evaluate all available agents
            for agent in self.agents:
                score, rationale = self.calculate_agent_score(agent, ticket)

                if score > best_score:
                    best_score = score
                    best_agent = agent
                    best_rationale = rationale

            if best_agent:
                # Create assignment
                assignment = Assignment(
                    ticket_id=ticket.ticket_id,
                    title=ticket.title,
                    assigned_agent_id=best_agent.agent_id,
                    rationale=best_rationale,
                    confidence_score=best_score
                )
                assignments.append(assignment)

                # Update load balancer
                self.load_balancer.assign_ticket(best_agent.agent_id)

        return assignments

    def generate_output(self, assignments: List[Assignment], output_path: str):
        """Generate the required output JSON file."""
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
        """Print assignment statistics for analysis."""
        print("\n" + "="*80)
        print("INTELLIGENT TICKET ASSIGNMENT SYSTEM - STATISTICS")
        print("="*80)

        # Agent workload distribution
        agent_assignments = defaultdict(int)
        agent_names = {agent.agent_id: agent.name for agent in self.agents}

        for assignment in assignments:
            agent_assignments[assignment.assigned_agent_id] += 1

        print("\nAgent Workload Distribution:")
        print("-" * 40)
        for agent in self.agents:
            original_load = agent.current_load
            new_assignments = agent_assignments[agent.agent_id]
            total_load = original_load + new_assignments
            print(
                f"{agent.name:20} | Original: {original_load:2d} | New: {new_assignments:2d} | Total: {total_load:2d}")

        # Priority distribution
        priority_distribution = defaultdict(int)
        for ticket in self.tickets:
            priority_distribution[ticket.urgency_level] += 1

        print("\nTicket Priority Distribution:")
        print("-" * 30)
        for priority, count in priority_distribution.items():
            print(f"{priority.capitalize():10} | {count:3d} tickets")

        # Skill coverage analysis
        skill_usage = defaultdict(int)
        for ticket in self.tickets:
            for skill in ticket.required_skills or []:
                skill_usage[skill] += 1

        print("\nTop Required Skills:")
        print("-" * 25)
        sorted_skills = sorted(skill_usage.items(),
                               key=lambda x: x[1], reverse=True)
        for skill, count in sorted_skills[:10]:
            print(f"{skill:25} | {count:3d} tickets")

        print("\n" + "="*80)


def main():
    """Main function to run the ticket assignment system."""
    print("Intelligent Support Ticket Assignment System")
    print("PyCon25 Hackathon Project")
    print("=" * 50)

    # Initialize the system
    system = TicketAssignmentSystem('dataset.json')

    print(
        f"\nLoaded {len(system.agents)} agents and {len(system.tickets)} tickets")
    print("Processing assignments...")

    # Generate assignments
    assignments = system.assign_tickets()

    # Generate output file
    system.generate_output(assignments, 'output_result.json')

    # Print statistics
    system.print_statistics(assignments)

    print(f"\nAssignment complete! Results saved to 'output_result.json'")
    print(f"Total assignments: {len(assignments)}")


if __name__ == "__main__":
    main()
