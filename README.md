to start the LangGraph Server Visit [This URL](#https://docs.langchain.com/langsmith/studio#local-development-server)

**Prerequisites**:
 - LangSmith [LangSmith Package Install](#https://docs.langchain.com/langsmith/home) and follow the instructions 
 - Agent Server [this](#https://docs.langchain.com/langsmith/agent-server#parts-of-a-deployment) 
 - [LangGraph CLI](#https://docs.langchain.com/langsmith/cli): 
    **Installation**: 
     1. Ensure Docker is installed (e.g., `docker --version`).
     2. Install the CLI:
        ```python
        pip install langgraph-cli 
        ```
    3. Verify the install:
        ```python
        langgraph --help
        ```
    
    4. To start the dev server:
    ```python
    pip install -U "langgraph-cli[inmem]"
    ```

     ```bash
    langgraph dev [OPTIONS]
    ```

