# lcel-rag-enterprise

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-LangChain%20v0.3-orange.svg)](https://python.langchain.com/)
[![Vector Database](https://img.shields.io/badge/vector%20db-Chroma-red.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, modular Retrieval-Augmented Generation (RAG) scaffolding engine built with LangChain Expression Language (LCEL) and Google Gemini. This repository serves as a deterministic production blueprint for closed-domain document processing, featuring strict type-safe structured outputs and high-availability retrieval fault isolation.

## 🚀 User Perspective
Most RAG tutorials and boilerplate applications rely on fragile, high-level sequential chains. While these work well for simple demonstrations, they rapidly degrade in production environments. We frequently observe architectures cracking under stress due to:
1. **Schema Drift:** LLMs failing to adhere to raw-text JSON prompt requests, breaking downstream microservices.
2. **Cascading Failures:** A single timeout or minor glitch in the vector database layer causing an unhandled exception that crashes the entire application thread.
3. **Monolithic Spaghetti:** Ingestion logic, prompt optimization, and pipeline orchestration crammed into single, non-testable scripts.

This repository was motivated by the need for a **hardened design pattern**—treating RAG infrastructure as a deterministic, fault-tolerant Directed Acyclic Graph (DAG) using low-level LangChain Expression Language (LCEL).

### Key Benefits
* **Decoupled Architecture:** Strict separation of concerns across document ingestion, retrieval tools, system prompts, and pipeline topologies.
* **Deterministic Structured Outputs:** Replaces unstable raw-text JSON prompt instructions with compile-time Pydantic validation mapped directly to Gemini's native structured inference engine.
* **Fault-Tolerant Circuit Breaking:** Features a resilient retrieval architecture using LCEL `.with_fallbacks()` mechanics to guarantee system availability even during transient vector database drops.
* **Parallelized Flow DAGs:** Maximizes runtime efficiency by processing context retrieval and query pass-throughs concurrently via asynchronous execution blocks.

---

## 📂 Repository Layout & Architecture

The codebase strictly isolates logical operations to ensure clean code maintenance, comprehensive unit testing, and isolated subsystem updates:

```text
gemini-lcel-rag-enterprise/
├── .env.example            # Baseline environment configurations template
├── main.py                 # System orchestrator / execution root script
├── ingestion/              # Ingestion Hub
│   ├── __init__.py
│   └── indexer.py          # Document loader, token splitter, DB serialization
├── tools/                  # Extensible Interfaces
│   ├── __init__.py
│   └── retriever.py        # Vector query abstraction and fallback setup
├── prompts/                # Guardrailed Prompts
│   ├── __init__.py
│   └── rag_prompts.py      # Declarative context injection prompt layers
└── chains/                 # Execution DAG Layouts
    ├── __init__.py
    └── rag_chain.py        # Pipeline design built strictly via LCEL syntax

🤝 Where Users Can Get Help
Issue Tracker: Please log system bugs, environment issues, or architectural feature requests via our GitHub Issues page.

Documentation Workspace: Detailed operational blueprints, database schema deep-dives, and performance scaling reports can be reviewed inside the local docs/ directory.

Discussions Board: For general architectural questions, orchestration strategy advice, or community integrations, join our public discussion forums.

👥 Who Maintains and Contributes
Maintainers
Core Engineering Team — Initial Architecture & System Maintenance — GitHub Profile

Contributing
We welcome optimizations to the execution graph, alternative vector store integrations, and performance enhancements. Please read our local docs/CONTRIBUTING.md handbook for precise details regarding code formatting standards, branch configurations, and standard pull request evaluation workflows.

License
This enterprise repository is open-source software licensed under the terms of the MIT License. Review the LICENSE file located in the project root directory for full legal parameters
