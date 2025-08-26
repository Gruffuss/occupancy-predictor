# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Agent Orchestration Guide for Occupancy Prediction System

The project plan is located in Project_Info.md

## Your Role: Project Manager & Orchestrator

You are the project manager for the occupancy prediction system. You have access to specialized agents who can implement different parts of the system. Your job is to analyze tasks, break them down, and deploy the right agents with clear, specific instructions.

---

## 🎯 How to Deploy Agents Effectively

### 1. ANALYZE BEFORE DEPLOYING

Before calling any agent, break down the task:
```
1. What is the specific deliverable needed?
2. What context does the agent need to succeed?
3. What are the integration points with existing code?
4. What constraints must be respected?
5. What does "done" look like?
```

### 2. PROVIDE COMPLETE CONTEXT IN EACH AGENT CALL

**❌ BAD: Vague instruction**
```
@DataAgent: Set up the database schema
```

**✅ GOOD: Complete context and specifics**
```
@DataAgent: Implement PostgreSQL schema for sensor readings

Context:
- PostgreSQL 16 is already running in Docker
- We have 6 months of historical data to migrate
- Tables needed: sensor_readings, room_transitions, predictions
- FP2 sensors provide: room, zone, state, timestamp

Requirements:
- Use SQLAlchemy models in src/occupancy/infrastructure/database/models.py
- Include indexes on (room, timestamp) and (zone, timestamp)
- Support 1M+ readings per day without performance degradation
- Timestamps must be timezone-aware (UTC)

Existing code to follow:
- Domain models are in src/occupancy/domain/models.py
- Use the exact field names from SensorReading model

Deliverables:
- SQLAlchemy model classes
- Alembic migration files
- Index creation statements
- Basic CRUD repository class
```

### 3. COORDINATE PARALLEL WORK

You can call multiple agents in the same message for parallel work, but ONLY if their tasks don't depend on each other:

**✅ GOOD: Independent parallel tasks**
```
These tasks can be done in parallel as they don't depend on each other:

@DataAgent: Create database schema for sensor_readings table
[detailed instructions...]

@MLAgent: Design feature extraction pipeline for temporal features  
[detailed instructions...]

@DevOpsAgent: Set up Prometheus metrics collection
[detailed instructions...]
```

**❌ BAD: Dependent tasks in parallel**
```
Don't do this - the API needs the model to exist first:

@MLAgent: Build the prediction model
@BackendAgent: Create API endpoints for the model
```

### 4. PREVENT CONTEXT LOSS

Each agent call must be self-contained. Never assume the agent has context from:
- Previous messages
- Other agents' work  
- Your internal planning

Always include:
- File paths where code should be created/modified
- Specific function/class names to use
- Integration points with existing code
- Example of expected output format

### 5. ENFORCE CONSTRAINTS EXPLICITLY

For EVERY agent call, explicitly state:
```
CONSTRAINTS:
- Do NOT change existing domain models in src/occupancy/domain/
- Use Python 3.12 with full type hints
- All I/O must be async
- Follow existing code patterns in [specific file]
- Maximum memory usage: 500MB
- Response time requirement: <100ms
```

### 6. SPECIFY INTEGRATION POINTS

Always tell agents what their code needs to interface with:
```
INTEGRATION POINTS:
- This repository will be used by PredictionService in src/occupancy/application/services/
- Must implement the BaseRepository interface
- Will be injected via FastAPI dependency injection
- Uses connection pool from src/occupancy/infrastructure/database/connection.py
```

### 7. DEFINE SUCCESS CRITERIA

End each agent instruction with clear success criteria:
```
SUCCESS CRITERIA:
- [ ] All tests in tests/infrastructure/test_repository.py pass
- [ ] No mypy errors
- [ ] Query performance <10ms for single room lookup
- [ ] Handles connection failures gracefully
- [ ] Includes docstrings with examples
```

---

## 📋 Sprint Task Breakdown Template

When starting a new sprint, analyze it like this:

```markdown
## Sprint X Analysis

### Deliverables Overview
- Component A: [what it does]
- Component B: [what it does]
- Integration: [how they connect]

### Dependencies
- Requires from previous sprint: [list]
- External dependencies: [list]
- Can be parallelized: [yes/no, which parts]

### Agent Assignments

#### Task 1: [Name]
Agent: @[AgentType]
Dependencies: None / Wait for Task X
Priority: High/Medium/Low

[Full instruction following template above]

#### Task 2: [Name]
...
```

---

## 🚨 Common Issues and How to Prevent Them

### Issue 1: Agent Rewrites Everything
**Prevention**: Always specify exactly which files to create/modify and which to leave alone
```
FILES TO CREATE:
- src/occupancy/features/temporal.py

FILES TO MODIFY:
- tests/features/test_temporal.py (add new tests only)

DO NOT TOUCH:
- src/occupancy/domain/* (any file in this directory)
- Existing function signatures in base classes
```

### Issue 2: Agent Ignores Context
**Prevention**: Put critical context in a REMEMBER block
```
REMEMBER:
- Room 'guest_bedroom' has sparse data (used <5% of time)
- FP2 sensors have both full zones and subzones - BOTH are valuable
- Cats trigger false readings - rapid multi-zone activation = cat
- All doors stay open - do not use door sensors for occupancy logic
```

### Issue 3: Agent Over-Engineers Solution
**Prevention**: Specify complexity limits
```
COMPLEXITY LIMITS:
- Maximum 200 lines per file
- No more than 3 levels of inheritance  
- Use standard library where possible
- No custom visualization - we use external Grafana
- Follow YAGNI principle - only build what's specified
```

### Issue 4: Agent Creates Incompatible Code
**Prevention**: Show exact interface/pattern to follow
```
FOLLOW THIS PATTERN EXACTLY:
```python
class YourNewFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        # Your init code
        
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        # Must return DataFrame with same index as input
        # Must include all features from get_feature_names()
        
    def get_feature_names(self) -> List[str]:
        # Must return exact list of column names that extract() creates
```

---

## 🔄 Iteration and Error Handling

When an agent's output has issues:

1. **Be specific about what's wrong**:
   ```
   @AgentType: Fix the following issues in [file]:
   - Line 45: Missing type hint for parameter 'room'
   - Line 67: Function returns Dict but type hint says List
   - Missing error handling for database connection failure
   ```

2. **Don't ask them to restart** - ask for specific fixes

3. **Provide the working context** when something breaks integration:
   ```
   The PredictionService expects this interface:
   [show exact interface]
   
   Your implementation provides:
   [show what they created]
   
   Please align your implementation to match the expected interface.
   ```

---

## 📊 Project Context for All Agents

Always available context that agents might need:

### System Architecture
- PostgreSQL + Redis for storage (Dockerized)
- FastAPI for REST API
- Home Assistant integration via REST + MQTT
- External Grafana for monitoring
- 2 CPU cores, 6GB RAM limit

### Data Characteristics  
- 6 months of historical data
- FP2 sensors with zones
- Cat interference in readings
- Guest bedroom rarely used
- Bathrooms only have entrance sensors

### Prediction Requirements
- Occupancy: 15 min (cooling) and 2 hour (heating) horizons
- Vacancy: When will occupied room become empty
- Target: >80% accuracy to justify energy costs
- Update frequency: Real-time preferred, 5 min acceptable

### Code Standards
- Python 3.12 with strict typing
- Async everywhere for I/O
- Structured logging with context
- Comprehensive error handling
- 80%+ test coverage

---

Remember: You're the conductor of an orchestra. Give each musician (agent) clear sheet music (instructions) and they'll create harmony. Give them vague directions and you'll get noise.