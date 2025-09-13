#!/usr/bin/env python3
"""
Utility Functions for Intelligent Support Ticket Assignment System
Includes validation, analysis tools, and helper functions
"""

import json
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def validate_dataset(dataset_path: str) -> Tuple[bool, List[str]]:
    """
    Validate the input dataset for completeness and correctness.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return False, ["Dataset file not found"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON format: {e}"]

    # Check top-level structure
    if 'agents' not in data:
        issues.append("Missing 'agents' section")
    if 'tickets' not in data:
        issues.append("Missing 'tickets' section")

    if issues:
        return False, issues

    # Validate agents
    if not data['agents']:
        issues.append("No agents found in dataset")
    else:
        required_agent_fields = ['agent_id', 'name', 'skills',
                                 'current_load', 'availability_status', 'experience_level']
        for i, agent in enumerate(data['agents']):
            for field in required_agent_fields:
                if field not in agent:
                    issues.append(
                        f"Agent {i}: Missing required field '{field}'")

            # Validate skill scores
            if 'skills' in agent and isinstance(agent['skills'], dict):
                for skill, score in agent['skills'].items():
                    if not isinstance(score, int) or score < 0 or score > 10:
                        issues.append(
                            f"Agent {i}: Invalid skill score for '{skill}' (should be 0-10)")

    # Validate tickets
    if not data['tickets']:
        issues.append("No tickets found in dataset")
    else:
        required_ticket_fields = ['ticket_id',
                                  'title', 'description', 'creation_timestamp']
        for i, ticket in enumerate(data['tickets']):
            for field in required_ticket_fields:
                if field not in ticket:
                    issues.append(
                        f"Ticket {i}: Missing required field '{field}'")

            # Validate timestamp
            if 'creation_timestamp' in ticket:
                if not isinstance(ticket['creation_timestamp'], (int, float)):
                    issues.append(f"Ticket {i}: Invalid timestamp format")

    return len(issues) == 0, issues


def validate_output(output_path: str) -> Tuple[bool, List[str]]:
    """
    Validate the output file format and completeness.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    try:
        with open(output_path, 'r') as f:
            assignments = json.load(f)
    except FileNotFoundError:
        return False, ["Output file not found"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON format: {e}"]

    if not isinstance(assignments, list):
        return False, ["Output should be a list of assignments"]

    required_fields = ['ticket_id', 'assigned_agent_id']
    optional_fields = ['title', 'rationale']

    for i, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            issues.append(f"Assignment {i}: Should be a dictionary")
            continue

        for field in required_fields:
            if field not in assignment:
                issues.append(
                    f"Assignment {i}: Missing required field '{field}'")

        # Check ticket_id format
        if 'ticket_id' in assignment:
            if not isinstance(assignment['ticket_id'], str) or not assignment['ticket_id'].startswith('TKT-'):
                issues.append(f"Assignment {i}: Invalid ticket_id format")

        # Check agent_id format
        if 'assigned_agent_id' in assignment:
            if not isinstance(assignment['assigned_agent_id'], str) or not assignment['assigned_agent_id'].startswith('agent_'):
                issues.append(f"Assignment {i}: Invalid agent_id format")

    return len(issues) == 0, issues


def analyze_assignments(dataset_path: str, output_path: str) -> Dict[str, Any]:
    """
    Analyze assignment quality and provide detailed metrics.

    Returns:
        Dictionary with analysis results
    """
    # Load data
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'r') as f:
        assignments = json.load(f)

    # Create lookup dictionaries
    agents = {agent['agent_id']: agent for agent in data['agents']}
    tickets = {ticket['ticket_id']: ticket for ticket in data['tickets']}

    analysis = {
        'assignment_quality': {},
        'load_distribution': {},
        'skill_utilization': {},
        'priority_handling': {},
        'efficiency_metrics': {}
    }

    # Assignment Quality Analysis
    skill_matches = 0
    total_assignments = len(assignments)

    for assignment in assignments:
        ticket_id = assignment['ticket_id']
        agent_id = assignment['assigned_agent_id']

        if ticket_id in tickets and agent_id in agents:
            ticket = tickets[ticket_id]
            agent = agents[agent_id]

            # Simple skill matching check (could be enhanced)
            ticket_text = (ticket['title'] + ' ' +
                           ticket['description']).lower()
            agent_skills = list(agent['skills'].keys())

            has_match = any(skill.lower().replace('_', ' ')
                            in ticket_text for skill in agent_skills)
            if has_match:
                skill_matches += 1

    analysis['assignment_quality'] = {
        'skill_match_rate': skill_matches / total_assignments if total_assignments > 0 else 0,
        'total_assignments': total_assignments,
        'skilled_assignments': skill_matches
    }

    # Load Distribution Analysis
    agent_loads = defaultdict(int)
    for assignment in assignments:
        agent_loads[assignment['assigned_agent_id']] += 1

    original_loads = [agent['current_load'] for agent in data['agents']]
    new_loads = [agent_loads[agent['agent_id']] for agent in data['agents']]
    total_loads = [orig + new for orig, new in zip(original_loads, new_loads)]

    analysis['load_distribution'] = {
        'original_load_std': statistics.stdev(original_loads) if len(original_loads) > 1 else 0,
        'new_assignment_std': statistics.stdev(new_loads) if len(new_loads) > 1 else 0,
        'total_load_std': statistics.stdev(total_loads) if len(total_loads) > 1 else 0,
        'load_balance_score': 1 / (1 + statistics.stdev(total_loads)) if len(total_loads) > 1 else 1,
        'agent_utilization': dict(zip([agent['agent_id'] for agent in data['agents']], total_loads))
    }

    # Skill Utilization Analysis
    skill_usage = defaultdict(int)
    for agent in data['agents']:
        for skill in agent['skills']:
            if agent['agent_id'] in agent_loads:
                skill_usage[skill] += agent_loads[agent['agent_id']]

    analysis['skill_utilization'] = {
        'most_used_skills': dict(Counter(skill_usage).most_common(10)),
        'skill_diversity': len(skill_usage),
        'total_skill_instances': sum(skill_usage.values())
    }

    # Priority Handling (simplified analysis)
    critical_keywords = ['critical', 'urgent', 'emergency', 'down', 'outage']
    priority_assignments = 0

    for assignment in assignments:
        ticket_id = assignment['ticket_id']
        if ticket_id in tickets:
            ticket_text = (tickets[ticket_id]['title'] +
                           ' ' + tickets[ticket_id]['description']).lower()
            if any(keyword in ticket_text for keyword in critical_keywords):
                priority_assignments += 1

    analysis['priority_handling'] = {
        'priority_tickets_identified': priority_assignments,
        'priority_ratio': priority_assignments / total_assignments if total_assignments > 0 else 0
    }

    # Efficiency Metrics
    agent_experience_utilization = 0
    for assignment in assignments:
        agent_id = assignment['assigned_agent_id']
        if agent_id in agents:
            agent_experience_utilization += agents[agent_id]['experience_level']

    avg_experience_utilization = agent_experience_utilization / \
        total_assignments if total_assignments > 0 else 0

    analysis['efficiency_metrics'] = {
        'average_experience_utilization': avg_experience_utilization,
        'assignments_per_agent': total_assignments / len(data['agents']) if len(data['agents']) > 0 else 0,
        'coverage_rate': len(set(assignment['assigned_agent_id'] for assignment in assignments)) / len(data['agents']) if len(data['agents']) > 0 else 0
    }

    return analysis


def benchmark_system_performance(dataset_path: str, iterations: int = 5) -> Dict[str, float]:
    """
    Benchmark system performance over multiple iterations.

    Returns:
        Dictionary with performance metrics
    """
    from intelligent_assignment_system import TicketAssignmentSystem

    times = []
    memory_usage = []

    for i in range(iterations):
        start_time = time.time()

        # Initialize system
        system = TicketAssignmentSystem(dataset_path)

        # Generate assignments
        assignments = system.assign_tickets()

        end_time = time.time()
        times.append(end_time - start_time)

        # Simple memory estimation (not exact)
        import sys
        memory_est = sys.getsizeof(
            system.agents) + sys.getsizeof(system.tickets) + sys.getsizeof(assignments)
        memory_usage.append(memory_est)

    return {
        'avg_execution_time': statistics.mean(times),
        'min_execution_time': min(times),
        'max_execution_time': max(times),
        'std_execution_time': statistics.stdev(times) if len(times) > 1 else 0,
        'avg_memory_usage': statistics.mean(memory_usage),
        'assignments_per_second': len(assignments) / statistics.mean(times) if statistics.mean(times) > 0 else 0
    }


def generate_assignment_report(dataset_path: str, output_path: str, report_path: str = 'assignment_report.json'):
    """
    Generate comprehensive assignment report with analysis and metrics.
    """
    print("📊 Generating comprehensive assignment report...")

    # Validate input and output
    dataset_valid, dataset_issues = validate_dataset(dataset_path)
    output_valid, output_issues = validate_output(output_path)

    # Analyze assignments
    analysis = analyze_assignments(
        dataset_path, output_path) if dataset_valid and output_valid else {}

    # Benchmark performance
    try:
        performance = benchmark_system_performance(dataset_path, iterations=3)
    except Exception as e:
        performance = {'error': str(e)}

    # Create comprehensive report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'validation': {
            'dataset_valid': dataset_valid,
            'dataset_issues': dataset_issues,
            'output_valid': output_valid,
            'output_issues': output_issues
        },
        'analysis': analysis,
        'performance': performance,
        'summary': {
            'system_status': 'healthy' if dataset_valid and output_valid else 'issues_detected',
            'recommendation': 'Production ready' if dataset_valid and output_valid and not analysis.get('assignment_quality', {}).get('skill_match_rate', 0) < 0.8 else 'Needs review'
        }
    }

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Report saved to {report_path}")

    # Print summary
    print("\n📋 Assignment Report Summary")
    print("-" * 40)

    if dataset_valid and output_valid:
        print(f"✅ System Status: {report['summary']['system_status']}")
        print(
            f"🎯 Skill Match Rate: {analysis.get('assignment_quality', {}).get('skill_match_rate', 0):.1%}")
        print(
            f"⚖️  Load Balance Score: {analysis.get('load_distribution', {}).get('load_balance_score', 0):.3f}")
        print(
            f"⚡ Processing Speed: {performance.get('assignments_per_second', 0):.1f} assignments/sec")
        print(f"💡 Recommendation: {report['summary']['recommendation']}")
    else:
        print("❌ System has validation issues")
        if dataset_issues:
            print("Dataset Issues:", dataset_issues)
        if output_issues:
            print("Output Issues:", output_issues)

    return report


def compare_assignments(output1_path: str, output2_path: str) -> Dict[str, Any]:
    """
    Compare two assignment outputs to analyze differences.

    Returns:
        Dictionary with comparison results
    """
    with open(output1_path, 'r') as f:
        assignments1 = json.load(f)

    with open(output2_path, 'r') as f:
        assignments2 = json.load(f)

    # Create lookup dictionaries
    lookup1 = {assignment['ticket_id']: assignment['assigned_agent_id']
               for assignment in assignments1}
    lookup2 = {assignment['ticket_id']: assignment['assigned_agent_id']
               for assignment in assignments2}

    # Find differences
    all_tickets = set(lookup1.keys()) | set(lookup2.keys())
    differences = []
    matches = 0

    for ticket_id in all_tickets:
        agent1 = lookup1.get(ticket_id, 'MISSING')
        agent2 = lookup2.get(ticket_id, 'MISSING')

        if agent1 == agent2:
            matches += 1
        else:
            differences.append({
                'ticket_id': ticket_id,
                'assignment1': agent1,
                'assignment2': agent2
            })

    return {
        'total_tickets': len(all_tickets),
        'matching_assignments': matches,
        'different_assignments': len(differences),
        'agreement_rate': matches / len(all_tickets) if all_tickets else 0,
        'differences': differences[:10]  # Show first 10 differences
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utilities.py <command> [args]")
        print("Commands:")
        print("  validate_dataset <dataset.json>")
        print("  validate_output <output.json>")
        print("  analyze <dataset.json> <output.json>")
        print("  benchmark <dataset.json>")
        print("  report <dataset.json> <output.json>")
        print("  compare <output1.json> <output2.json>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate_dataset" and len(sys.argv) == 3:
        valid, issues = validate_dataset(sys.argv[2])
        print(f"Dataset valid: {valid}")
        if issues:
            print("Issues:", issues)

    elif command == "validate_output" and len(sys.argv) == 3:
        valid, issues = validate_output(sys.argv[2])
        print(f"Output valid: {valid}")
        if issues:
            print("Issues:", issues)

    elif command == "analyze" and len(sys.argv) == 4:
        analysis = analyze_assignments(sys.argv[2], sys.argv[3])
        print(json.dumps(analysis, indent=2))

    elif command == "benchmark" and len(sys.argv) == 3:
        performance = benchmark_system_performance(sys.argv[2])
        print(json.dumps(performance, indent=2))

    elif command == "report" and len(sys.argv) == 4:
        generate_assignment_report(sys.argv[2], sys.argv[3])

    elif command == "compare" and len(sys.argv) == 4:
        comparison = compare_assignments(sys.argv[2], sys.argv[3])
        print(json.dumps(comparison, indent=2))

    else:
        print("Invalid command or arguments")
        sys.exit(1)
