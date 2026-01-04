# self-hosted-coding-agent

CLI utilities for orchestrating a self-hosted coding agent that runs on RunPod, keeps chat context on your local machine, and injects your `.context` directory into the model prompt through lightweight RAG.

## Features

- Spin up and tear down RunPod AI workstations from your terminal.
- Local-first chat logs and RAG index; nothing is uploaded except what you explicitly send to the model.
- Default MCP server definitions for local terminal and filesystem control.
- Automatic idle shutdown window (30 minutes by default) so pods do not stay alive when unused.

## Quickstart

1. Install dependencies with Poetry (Python 3.11+):

   ```bash
   poetry install
   ```

2. Create a `.context` directory with any repo notes you want in the model prompt.

3. Initialize configuration (stores state in `.agent_state/`):

   ```bash
   poetry run self-hosted-agent config
   ```

   You will be prompted for your RunPod API key and optional template ID/image/machine type.

4. Build the local RAG index from `.context`:

   ```bash
   poetry run self-hosted-agent index
   ```

5. Launch a workstation:

   ```bash
   poetry run self-hosted-agent launch
   ```

6. Chat with the model (retrieval enabled by default):

   ```bash
   poetry run self-hosted-agent chat "Summarize the repository layout"
   ```

7. Periodically run the monitor to enforce idle shutdown (30-minute default):

   ```bash
   poetry run self-hosted-agent monitor
   ```

8. Terminate the RunPod pod manually when finished:

   ```bash
   poetry run self-hosted-agent shutdown
   ```

## Notes

- MCP server configuration is written to `.agent_state/mcp-servers.json` and targets local shell and filesystem servers.
- Chat history is stored locally in `.agent_state/chat_history.jsonl`.
- The agent assumes a model endpoint compatible with the OpenAI Chat Completions API (for example, an open-source model served on the RunPod workstation). Update `model_endpoint` and `model_name` in `.agent_state/config.json` as needed.
