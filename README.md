# Web Deep Research

An advanced web research automation system that combines Large Language Models (LLMs) with browser automation to conduct thorough, validated research with multiple source verification.

## Features

- **Automated Research Planning**: Utilizes a planning agent to strategically guide the research process
- **Multi-Source Validation**: Verifies information across multiple independent sources
- **Credibility Assessment**: Implements a scoring system for source reliability
- **Contradiction Detection**: Automatically identifies and resolves conflicting information
- **Dynamic Route Decision**: Adapts research strategy based on verification results
- **Memory Management**: Maintains verified facts, pending verifications, and contradiction tracking

## System Architecture

The system consists of two main components:

1. **Planning Agent**: Manages the research strategy and validation process
   - Validates browser results
   - Makes dynamic routing decisions
   - Maintains a memory system for tracking verified facts
   - Ensures comprehensive source verification

2. **Browser Agent**: Handles web interactions and data collection
   - Executes search queries
   - Extracts relevant information
   - Interfaces with the planning agent

## Requirements

- Python 3.x
- `langchain-openai`
- OpenAI API access (GPT-4 recommended)
- Browser automation dependencies (specified in browser_use.py)

## Usage

```python
import asyncio
from browser_use_search_agent import main

asyncio.run(main())
```

Example research query:

```python
"Open the browser and search for: 'Animals mentioned in Ilias Lagkouvardos and Olga Tapia papers on the alvei species that also appear in the 2021 Wikipedia article about a multicenter, randomized, double-blind study.'"
```

## Research Process
1. Initial Search : Gathers basic information framework
2. Validation Phase :
   - Cross-checks sources
   - Performs freshness checks (preferring recent studies)
   - Conducts reverse validation
3. Final Confirmation : Ensures all information is:
   - Verified by at least two independent sources
   - Free of major contradictions
   - Sufficiently detailed
## Output Format
The system provides structured output including:

- Comprehensive verified sources
- Final conclusions with fact assertions
- Source citations with years
- Validation status for each piece of information
## Time Management
- Default maximum execution time: 300 seconds (5 minutes)
- Automatic termination with summary if time limit is reached
