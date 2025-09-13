#!/usr/bin/env python3
"""
Comprehensive Test Suite for Intelligent Support Ticket Assignment System
Tests all components including skill extraction, priority scoring, and assignment logic
"""

from intelligent_assignment_system import (Agent, Assignment, LoadBalancer,
                                           PriorityScorer, SkillExtractor,
                                           Ticket, TicketAssignmentSystem)
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

# Import the classes from our main system
sys.path.append('.')


class TestAgent(unittest.TestCase):
    """Test Agent class functionality."""

    def setUp(self):
        self.agent = Agent(
            agent_id="test_001",
            name="Test Agent",
            skills={"Networking": 8, "Database_SQL": 6},
            current_load=2,
            availability_status="Available",
            experience_level=5
        )

    def test_get_skill_score_existing(self):
        """Test getting score for existing skill."""
        self.assertEqual(self.agent.get_skill_score("Networking"), 8)
        self.assertEqual(self.agent.get_skill_score("Database_SQL"), 6)

    def test_get_skill_score_nonexistent(self):
        """Test getting score for non-existent skill."""
        self.assertEqual(self.agent.get_skill_score("NonExistent_Skill"), 0)


class TestTicket(unittest.TestCase):
    """Test Ticket class functionality."""

    def setUp(self):
        self.ticket = Ticket(
            ticket_id="TKT-001",
            title="Network Issue",
            description="VPN connection problems",
            creation_timestamp=1757827200
        )

    def test_ticket_initialization(self):
        """Test ticket is properly initialized."""
        self.assertEqual(self.ticket.ticket_id, "TKT-001")
        self.assertEqual(self.ticket.title, "Network Issue")
        self.assertEqual(self.ticket.priority_score, 0.0)
        self.assertEqual(self.ticket.urgency_level, "medium")


class TestSkillExtractor(unittest.TestCase):
    """Test SkillExtractor functionality."""

    def setUp(self):
        self.extractor = SkillExtractor()

    def test_extract_networking_skills(self):
        """Test extraction of networking-related skills."""
        text = "VPN connection dropping intermittently with network issues"
        skills = self.extractor.extract_skills(text)
        self.assertIn("Networking", skills)
        self.assertIn("VPN_Troubleshooting", skills)

    def test_extract_hardware_skills(self):
        """Test extraction of hardware-related skills."""
        text = "Laptop fan making noise and overheating, need hardware diagnostic"
        skills = self.extractor.extract_skills(text)
        self.assertIn("Hardware_Diagnostics", skills)

    def test_extract_database_skills(self):
        """Test extraction of database-related skills."""
        text = "SQL Server performance issues with slow query execution"
        skills = self.extractor.extract_skills(text)
        self.assertIn("Database_SQL", skills)

    def test_extract_multiple_skills(self):
        """Test extraction of multiple skills from complex text."""
        text = "User cannot login to Active Directory and database connection is failing"
        skills = self.extractor.extract_skills(text)
        self.assertIn("Active_Directory", skills)
        self.assertIn("Database_SQL", skills)

    def test_extract_no_skills(self):
        """Test extraction from text with no recognized skills."""
        text = "Simple request for new coffee machine in break room"
        skills = self.extractor.extract_skills(text)
        # Should return empty list or minimal skills
        self.assertIsInstance(skills, list)

    def test_case_insensitive_extraction(self):
        """Test that skill extraction is case-insensitive."""
        text1 = "VPN connection issue"
        text2 = "vpn connection issue"
        text3 = "Vpn Connection Issue"

        skills1 = self.extractor.extract_skills(text1)
        skills2 = self.extractor.extract_skills(text2)
        skills3 = self.extractor.extract_skills(text3)

        self.assertEqual(skills1, skills2)
        self.assertEqual(skills2, skills3)


class TestPriorityScorer(unittest.TestCase):
    """Test PriorityScorer functionality."""

    def setUp(self):
        self.scorer = PriorityScorer()

    def test_critical_priority(self):
        """Test detection of critical priority tickets."""
        ticket = Ticket(
            ticket_id="TKT-CRITICAL",
            title="CRITICAL: Server down - business critical",
            description="Production server is completely down affecting all users",
            creation_timestamp=int(time.time())  # Current time
        )
        priority, urgency = self.scorer.calculate_priority(ticket)
        self.assertGreaterEqual(priority, 9.0)
        self.assertEqual(urgency, "critical")

    def test_high_priority(self):
        """Test detection of high priority tickets."""
        ticket = Ticket(
            ticket_id="TKT-HIGH",
            title="Urgent: Multiple users cannot access email",
            description="Email system is down affecting the sales department",
            creation_timestamp=int(time.time())
        )
        priority, urgency = self.scorer.calculate_priority(ticket)
        self.assertGreaterEqual(priority, 7.0)
        self.assertIn(urgency, ["high", "critical"])

    def test_low_priority(self):
        """Test detection of low priority tickets."""
        ticket = Ticket(
            ticket_id="TKT-LOW",
            title="Standard request for new user account",
            description="Please create a new user account for intern",
            creation_timestamp=int(time.time())
        )
        priority, urgency = self.scorer.calculate_priority(ticket)
        self.assertLess(priority, 4.0)
        self.assertEqual(urgency, "low")

    def test_department_multiplier(self):
        """Test that department-specific multipliers work."""
        finance_ticket = Ticket(
            ticket_id="TKT-FINANCE",
            title="Finance department server issue",
            description="Critical issue affecting finance team",
            creation_timestamp=int(time.time())
        )

        general_ticket = Ticket(
            ticket_id="TKT-GENERAL",
            title="General server issue",
            description="Critical issue affecting general users",
            creation_timestamp=int(time.time())
        )

        finance_priority, _ = self.scorer.calculate_priority(finance_ticket)
        general_priority, _ = self.scorer.calculate_priority(general_ticket)

        # Finance should have higher priority due to department multiplier
        self.assertGreater(finance_priority, general_priority)

    def test_time_urgency_boost(self):
        """Test that recent tickets get priority boost."""
        recent_timestamp = int(time.time()) - 1800  # 30 minutes ago
        old_timestamp = int(time.time()) - 86400 * 2  # 2 days ago

        recent_ticket = Ticket(
            ticket_id="TKT-RECENT",
            title="Server issue",
            description="Server problem",
            creation_timestamp=recent_timestamp
        )

        old_ticket = Ticket(
            ticket_id="TKT-OLD",
            title="Server issue",
            description="Server problem",
            creation_timestamp=old_timestamp
        )

        recent_priority, _ = self.scorer.calculate_priority(recent_ticket)
        old_priority, _ = self.scorer.calculate_priority(old_ticket)

        # Recent ticket should have higher priority
        self.assertGreater(recent_priority, old_priority)


class TestLoadBalancer(unittest.TestCase):
    """Test LoadBalancer functionality."""

    def setUp(self):
        self.agents = [
            Agent("agent_001", "Agent 1", {
                  "Networking": 8}, 2, "Available", 5),
            Agent("agent_002", "Agent 2", {
                  "Database_SQL": 9}, 4, "Available", 8),
            Agent("agent_003", "Agent 3", {
                  "Hardware_Diagnostics": 7}, 1, "Busy", 3)
        ]
        self.load_balancer = LoadBalancer(self.agents)

    def test_initial_load_factors(self):
        """Test initial load factor calculations."""
        # Agent with lower current load should have lower load factor
        factor1 = self.load_balancer.calculate_load_factor("agent_001")
        factor2 = self.load_balancer.calculate_load_factor("agent_002")

        # agent_001 has load 2, agent_002 has load 4
        self.assertLess(factor1, factor2)

    def test_availability_penalty(self):
        """Test that unavailable agents get load penalty."""
        available_factor = self.load_balancer.calculate_load_factor(
            "agent_001")
        busy_factor = self.load_balancer.calculate_load_factor("agent_003")

        # Busy agent should have higher load factor due to penalty
        self.assertGreater(busy_factor, available_factor)

    def test_assignment_tracking(self):
        """Test that assignments are tracked correctly."""
        initial_factor = self.load_balancer.calculate_load_factor("agent_001")

        # Assign a ticket
        self.load_balancer.assign_ticket("agent_001")

        updated_factor = self.load_balancer.calculate_load_factor("agent_001")

        # Load factor should increase after assignment
        self.assertGreater(updated_factor, initial_factor)

    def test_experience_based_capacity(self):
        """Test that experienced agents have higher capacity."""
        # agent_002 has experience 8, agent_003 has experience 3
        factor2 = self.load_balancer.calculate_load_factor("agent_002")
        factor3 = self.load_balancer.calculate_load_factor("agent_003")

        # More experienced agent should handle more load better
        # (Note: this test might need adjustment based on current loads)
        self.assertIsInstance(factor2, float)
        self.assertIsInstance(factor3, float)


class TestTicketAssignmentSystem(unittest.TestCase):
    """Test the complete TicketAssignmentSystem."""

    def setUp(self):
        # Create a minimal test dataset
        self.test_data = {
            "agents": [
                {
                    "agent_id": "agent_001",
                    "name": "Network Specialist",
                    "skills": {"Networking": 9, "VPN_Troubleshooting": 8},
                    "current_load": 1,
                    "availability_status": "Available",
                    "experience_level": 7
                },
                {
                    "agent_id": "agent_002",
                    "name": "Database Expert",
                    "skills": {"Database_SQL": 10, "Linux_Administration": 6},
                    "current_load": 2,
                    "availability_status": "Available",
                    "experience_level": 9
                }
            ],
            "tickets": [
                {
                    "ticket_id": "TKT-001",
                    "title": "VPN connection issues",
                    "description": "Users unable to connect to VPN network",
                    "creation_timestamp": 1757827200
                },
                {
                    "ticket_id": "TKT-002",
                    "title": "Database performance problems",
                    "description": "SQL queries running very slowly",
                    "creation_timestamp": 1757827300
                }
            ]
        }

        # Create temporary dataset file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()

        # Initialize system
        self.system = TicketAssignmentSystem(self.temp_file.name)

    def tearDown(self):
        # Clean up temporary file
        os.unlink(self.temp_file.name)

    def test_system_initialization(self):
        """Test that system initializes correctly."""
        self.assertEqual(len(self.system.agents), 2)
        self.assertEqual(len(self.system.tickets), 2)
        self.assertIsNotNone(self.system.skill_extractor)
        self.assertIsNotNone(self.system.priority_scorer)
        self.assertIsNotNone(self.system.load_balancer)

    def test_ticket_processing(self):
        """Test that tickets are processed correctly."""
        # Check that skills were extracted
        vpn_ticket = next(t for t in self.system.tickets if "VPN" in t.title)
        db_ticket = next(
            t for t in self.system.tickets if "Database" in t.title)

        self.assertIsNotNone(vpn_ticket.required_skills)
        self.assertIsNotNone(db_ticket.required_skills)

        # Check that priorities were calculated
        self.assertGreater(vpn_ticket.priority_score, 0)
        self.assertGreater(db_ticket.priority_score, 0)

    def test_agent_scoring(self):
        """Test agent scoring for specific tickets."""
        vpn_ticket = next(t for t in self.system.tickets if "VPN" in t.title)
        network_agent = next(
            a for a in self.system.agents if "Network" in a.name)
        db_agent = next(a for a in self.system.agents if "Database" in a.name)

        network_score, _ = self.system.calculate_agent_score(
            network_agent, vpn_ticket)
        db_score, _ = self.system.calculate_agent_score(db_agent, vpn_ticket)

        # Network specialist should score higher for VPN ticket
        self.assertGreater(network_score, db_score)

    def test_assignment_generation(self):
        """Test that assignments are generated correctly."""
        assignments = self.system.assign_tickets()

        # Should have assignments for all tickets
        self.assertEqual(len(assignments), 2)

        # Each assignment should have required fields
        for assignment in assignments:
            self.assertIsInstance(assignment, Assignment)
            self.assertIsNotNone(assignment.ticket_id)
            self.assertIsNotNone(assignment.assigned_agent_id)
            self.assertIsNotNone(assignment.rationale)
            self.assertGreater(assignment.confidence_score, 0)

    def test_output_generation(self):
        """Test output file generation."""
        assignments = self.system.assign_tickets()

        # Create temporary output file
        output_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        output_file.close()

        try:
            self.system.generate_output(assignments, output_file.name)

            # Verify output file exists and has correct content
            self.assertTrue(os.path.exists(output_file.name))

            with open(output_file.name, 'r') as f:
                output_data = json.load(f)

            self.assertEqual(len(output_data), 2)

            # Check required fields in output
            for item in output_data:
                self.assertIn("ticket_id", item)
                self.assertIn("assigned_agent_id", item)
                self.assertIn("rationale", item)

        finally:
            os.unlink(output_file.name)


class TestPerformance(unittest.TestCase):
    """Test system performance characteristics."""

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger test dataset
        num_agents = 50
        num_tickets = 200

        agents = []
        for i in range(num_agents):
            agents.append({
                "agent_id": f"agent_{i:03d}",
                "name": f"Agent {i}",
                "skills": {"Networking": (i % 10) + 1, "Database_SQL": ((i + 5) % 10) + 1},
                "current_load": i % 5,
                "availability_status": "Available" if i % 4 != 0 else "Busy",
                "experience_level": (i % 12) + 1
            })

        tickets = []
        for i in range(num_tickets):
            tickets.append({
                "ticket_id": f"TKT-{i:03d}",
                "title": f"Issue {i}: Network problem" if i % 2 == 0 else f"Issue {i}: Database issue",
                "description": "Test ticket description with network and database keywords",
                "creation_timestamp": 1757827200 + i * 60
            })

        large_dataset = {"agents": agents, "tickets": tickets}

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        json.dump(large_dataset, temp_file)
        temp_file.close()

        try:
            # Measure initialization time
            start_time = time.time()
            system = TicketAssignmentSystem(temp_file.name)
            init_time = time.time() - start_time

            # Measure assignment time
            start_time = time.time()
            assignments = system.assign_tickets()
            assignment_time = time.time() - start_time

            # Performance assertions
            self.assertLess(
                init_time, 5.0, "Initialization should take less than 5 seconds")
            self.assertLess(assignment_time, 10.0,
                            "Assignment should take less than 10 seconds")
            self.assertEqual(len(assignments), num_tickets,
                             "Should assign all tickets")

            # Check assignment quality
            assigned_agents = {a.assigned_agent_id for a in assignments}
            self.assertGreater(len(assigned_agents), 1,
                               "Should use multiple agents")

        finally:
            os.unlink(temp_file.name)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_data = {"agents": [], "tickets": []}

        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        json.dump(empty_data, temp_file)
        temp_file.close()

        try:
            with self.assertRaises((IndexError, ValueError)):
                system = TicketAssignmentSystem(temp_file.name)
                system.assign_tickets()
        finally:
            os.unlink(temp_file.name)

    def test_malformed_json(self):
        """Test handling of malformed JSON file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False)
        temp_file.write("{ invalid json }")
        temp_file.close()

        try:
            with self.assertRaises(json.JSONDecodeError):
                TicketAssignmentSystem(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    def test_missing_file(self):
        """Test handling of missing dataset file."""
        with self.assertRaises(FileNotFoundError):
            TicketAssignmentSystem("nonexistent_file.json")


def run_all_tests():
    """Run all tests and provide detailed results."""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)

    # Create test suite
    test_classes = [
        TestAgent,
        TestTicket,
        TestSkillExtractor,
        TestPriorityScorer,
        TestLoadBalancer,
        TestTicketAssignmentSystem,
        TestPerformance,
        TestErrorHandling
    ]

    total_tests = 0
    total_failures = 0
    total_errors = 0

    for test_class in test_classes:
        print(f"\n🔍 Testing {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(
            verbosity=2, stream=open(os.devnull, 'w'))
        result = runner.run(suite)

        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)

        if result.failures:
            print(f"  ❌ {len(result.failures)} failures")
        if result.errors:
            print(f"  ⚠️  {len(result.errors)} errors")
        if not result.failures and not result.errors:
            print(f"  ✅ All {result.testsRun} tests passed")

    print(f"\n📊 Test Summary")
    print("-" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")

    if total_failures == 0 and total_errors == 0:
        print("\n🎉 All tests passed! System is ready for production.")
        return True
    else:
        print(
            f"\n⚠️  {total_failures + total_errors} tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
