I want to build a proof-of-concept (POC) local MCP (Model Context Protocol) server worker using VS Code’s editor and Langgraph framework. The goal is to experiment with how Langgraph can be used to orchestrate interactions between different components – specifically, a local worker and a locally hosted language model from LM Studio.

**Here's the breakdown of what I need help with:**

1.  **MCP Server Worker Setup :**  I’ll start by implementing the basic structure outlined in the provided markdown file (research\_topics/agents/mcp\_server\_poc.md). This will involve setting up a minimal worker process that can receive and potentially respond to requests. I need guidance on structuring this initial POC.

2.  **Langgraph Integration:**  I want to use Langgraph to define the communication flow between the MCP server worker and other potential components (which, initially, will just be the LM Studio model).  Specifically, I’d like help understanding how to create Langgraph “nodes” that represent these interactions.

3.  **LM Studio Integration:** The core of this POC is connecting the MCP server worker to a language model running locally via LM Studio. I need guidance on:
    *   How to format requests and responses between the Langgraph nodes and the LM Studio API.
    *   How to handle data serialization/deserialization (e.g., JSON) for communication.

4.  **Poetry Dependency Management:**  I want to use Poetry to manage all project dependencies, including any necessary libraries for Langgraph, LM Studio interaction, and potentially the MCP server worker itself. Please provide a `pyproject.toml` file with appropriate dependencies.

5. **Minimal Working Example**: I'm aiming for a small, demonstrable example that showcases the core concepts rather than a fully-fledged production system.

**Specifically, can you:**

*   Provide step-by-step instructions on setting up the Langgraph environment in VS Code.
*   Offer code snippets illustrating how to define Langgraph nodes for communication with LM Studio (including request/response formatting).
*   Generate a `pyproject.toml` file using Poetry that includes necessary dependencies.
*   Suggest best practices for structuring this POC and scaling it if needed.
