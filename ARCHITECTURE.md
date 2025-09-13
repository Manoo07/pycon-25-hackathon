# Architecture Document: LLM-Powered Support Ticket Assignment System

## 1. Overview

The **Intelligent Support Ticket Assignment System** leverages **Retrieval-Augmented Generation (RAG)** combined with a **Large Language Model (LLM)** to analyze support tickets and assign them to the most suitable agents.

Incoming tickets are processed, rephrased, and embedded before performing similarity searches in a **vector database of agent profiles**. The LLM then reasons over the retrieved candidate agents to make the final assignment.

A dedicated **Agent Management Layer** ensures that the agent database is always up to date, reflecting new hires, updated skills, and availability changes. This enables fair workload distribution and accurate ticket routing at scale.

---

## 2. System Objectives

* **Accurate Assignment**: Route tickets to agents with the right expertise.
* **Dynamic Agent Management**: Automatically onboard new agents and update existing agent profiles in the vector database.
* **Fair Workload Distribution**: Continuously balance assignments as the agent pool changes.
* **Adaptive Reasoning**: Use LLMs to handle ambiguous or novel tickets.
* **Scalable Processing**: Efficiently handle large ticket volumes.
* **Explainability**: Provide assignment rationale for transparency and auditability.

---

## 3. High-Level Architecture

```
                  ┌───────────────────────┐
                  │   Ticket Ingestion     │
                  │  (API / Queue / File)  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Ticket Pre-Processor   │
                  │ - Cleaning             │
                  │ - Metadata extraction  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────────────┐
                  │  RAG Candidate Retrieval       │
                  │ - Rephrase ticket              │
                  │ - Embedding generation         │
                  │ - Vector DB search             │
                  │ - Return top 5 agents          │
                  └───────────┬───────────────────┘
                              │
                              ▼
                  ┌───────────────────────────────┐
                  │   LLM Assignment Module        │
                  │ - Few-shot prompting           │
                  │ - Reasoning over candidates    │
                  │ - Select best-fit agent        │
                  └───────────┬───────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
 ┌─────────────────────┐                 ┌─────────────────────┐
 │ Assignment Validator │                 │ Load Balancer       │
 │ - Skill validation   │                 │ - Equitable routing │
 │ - Availability check │                 │ - Escalation policy │
 └───────────┬─────────┘                 └───────────┬─────────┘
             │                                       │
             ▼                                       ▼
     ┌───────────────────────┐              ┌───────────────────────┐
     │ Output Generator       │              │ Monitoring & Feedback │
     │ (ticket → agent JSON) │              │ - Resolution metrics   │
     └───────────┬───────────┘              │ - Fairness reporting  │
                 │                          └───────────┬───────────┘
                 ▼                                      │
         ┌───────────────────────┐                     ▼
         │ output_result.json     │            ┌───────────────────────┐
         │ (assignments & logs)  │            │ Continuous Improvement │
         └───────────┬───────────┘            └───────────┬───────────┘
                     │                                    │
                     ▼                                    ▼
          ┌───────────────────────┐          ┌────────────────────────┐
          │ Agent Management Layer│          │ Agent Vector Database  │
          │ - Add/update agents   │◄────────►│ - Profiles, skills,    │
          │ - Re-embed profiles   │          │   proficiency vectors   │
          │ - Sync availability   │          │ - Used in RAG search   │
          └───────────────────────┘          └────────────────────────┘
```

---

## 4. Core Components

### 4.1 Ticket Ingestion

* Receives tickets via APIs, message queues, or batch files (`dataset.json`).
* Normalizes inputs to a standard schema.

### 4.2 Ticket Pre-Processor

* Cleans ticket descriptions and extracts metadata (timestamp, priority, category).
* Generates an enriched representation for downstream processing.

### 4.3 RAG Candidate Retrieval

* Rephrases ticket descriptions for clarity.
* Generates ticket embeddings using a pre-trained model.
* Queries the vector database for **top 5 candidate agents** based on skill, proficiency, and historical resolution success.

### 4.4 LLM Assignment Module

* Consumes the ticket and retrieved candidate agents.
* Uses **few-shot prompting** with historical examples.
* Selects the most suitable agent and provides justification.

### 4.5 Assignment Validator

* Ensures selected agent meets basic constraints:

  * Required skill set
  * Availability status
  * Workload thresholds

### 4.6 Load Balancer

* Monitors ticket distribution across agents.
* Ensures equitable workload sharing.
* Implements fallback escalation when no ideal candidate is available.

### 4.7 Output Generator

* Produces structured assignment results in `output_result.json`.
* Includes optional rationale for traceability.

### 4.8 Agent Management Layer

* Handles onboarding of **new agents** into the system.
* Updates skills, proficiency scores, and availability for existing agents.
* Re-generates embeddings and refreshes the vector database.
* Triggers rebalancing logic to incorporate new capacity.

### 4.9 Monitoring & Feedback

* Collects metrics on assignment accuracy, fairness, and resolution success.
* Captures agent and system feedback.
* Supports continuous improvement cycles.

---

## 5. Data Flow

1. Ticket received and pre-processed.
2. Ticket description rephrased and embedded.
3. Vector similarity search retrieves top 5 candidate agents.
4. LLM selects best-fit agent.
5. Assignment validated and balanced.
6. Assignment stored in `output_result.json`.
7. Monitoring system tracks resolution metrics.
8. Agent Management Layer continuously updates vector database with new/updated agents.

---

## 6. Success Metrics

* **Assignment Accuracy**: Percentage of tickets correctly resolved by the first assigned agent.
* **Workload Balance**: Variance in ticket distribution across agents.
* **Resolution Rate**: Percentage of tickets closed without reassignment.
* **Latency**: Average processing time per ticket.
* **Adaptability**: Time taken to integrate a new agent into routing.

---

## 7. Future Enhancements

* **Automated Skill Discovery**: Derive agent skill profiles from historical performance data.
* **Self-Healing Routing**: Automatically adjust vector embeddings based on resolution outcomes.
* **Advanced Prioritization**: Integrate SLA deadlines, customer importance, and escalation rules.
* **Multilingual Ticket Support**: Expand handling to non-English descriptions.

---

## 8. Agent Data Schema

### 8.1 Agent Profile Structure

```json
{
  "agent_id": "agent_001",
  "name": "Sarah Chen",
  "skills": {
    "Networking": 9,
    "Linux_Administration": 7,
    "Cloud_AWS": 5,
    "VPN_Troubleshooting": 8,
    "Hardware_Diagnostics": 3
  },
  "current_load": 2,
  "max_capacity": 15,
  "availability_status": "Available",
  "experience_level": 7,
  "specializations": ["Network Infrastructure", "Cloud Migration"],
  "language_skills": ["English", "Mandarin"],
  "certification_level": "Senior",
  "historical_performance": {
    "resolution_rate": 0.92,
    "avg_resolution_time": 4.5,
    "customer_satisfaction": 4.7
  },
  "schedule": {
    "timezone": "PST",
    "working_hours": "09:00-17:00",
    "on_call": false
  }
}
```

### 8.2 Dynamic Agent Management

The **Agent Management Layer** maintains this schema and ensures:

1. **New Agent Onboarding**:
   - Profile creation with skill assessment
   - Vector embedding generation
   - Database insertion
   - Capacity planning adjustment

2. **Existing Agent Updates**:
   - Skill level modifications
   - Availability status changes
   - Performance metric updates
   - Re-embedding and database refresh

3. **Agent Lifecycle Management**:
   - Temporary unavailability handling
   - Skill certification updates
   - Load balancing recalibration
   - Agent departure procedures

---

This architecture ensures that when **new agents are added**, they are immediately embedded into the vector database, making them part of future assignments. This keeps the system **adaptive, fair, and up to date**.
