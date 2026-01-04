# self-hosted-coding-agent

CLI utilities for orchestrating a self-hosted coding agent that runs on RunPod, keeps chat context on your local machine, and injects your `.context` directory into the model prompt through lightweight RAG.

## Features

- Spin up and tear down RunPod AI workstations from your terminal.
- Local-first chat logs and RAG index; nothing is uploaded except what you explicitly send to the model.
- Default MCP server definitions for local terminal and filesystem control.
- Automatic idle shutdown window (30 minutes by default) so pods do not stay alive when unused.
- High-quality defaults for large codebases: larger RAG chunks with overlap, more retrieved neighbors, low-temperature generation, and a guiding system prompt that encourages grounded, stepwise answers.

## Quickstart

1. Install dependencies with Poetry (Python 3.11+):

   ```bash
   poetry install
   ```

2. Create a `.context` directory with any repo notes you want in the model prompt.

3. Add your RunPod API key to a local `.env` file:

   ```
   RunPodAPIKey=YOUR_KEY_HERE
   ```

4. Initialize configuration (stores state in `.agent_state/`):

   ```bash
   poetry run self-hosted-agent config
   ```

   You will be prompted for your RunPod API key and optional template ID/image/machine type.

5. Build the local RAG index from `.context`:

   ```bash
   poetry run self-hosted-agent index
   ```

6. Launch a workstation:

   ```bash
   poetry run self-hosted-agent launch
   ```

7. Chat with the model (retrieval enabled by default):

   ```bash
   poetry run self-hosted-agent chat "Summarize the repository layout"
   ```

8. Periodically run the monitor to enforce idle shutdown (30-minute default):

   ```bash
   poetry run self-hosted-agent monitor
   ```

9. Terminate the RunPod pod manually when finished:

   ```bash
   poetry run self-hosted-agent shutdown
   ```

## Notes

- MCP server configuration is written to `.agent_state/mcp-servers.json` and targets local shell and filesystem servers.
- Chat history is stored locally in `.agent_state/chat_history.jsonl`.
- The agent assumes a model endpoint compatible with the OpenAI Chat Completions API (for example, an open-source model served on the RunPod workstation). Update `model_endpoint` and `model_name` in `.agent_state/config.json` as needed.
- Quality-oriented defaults (tunable in `.agent_state/config.json`):
  - `model_name`: `qwen2.5-coder-32b-instruct` (strong reasoning for large repos).
  - `system_prompt`: pushes grounded, stepwise answers with file-path citations.
  - `temperature`/`top_p` set low (0.2/0.9) for focused outputs; `max_tokens` 2048 for longer replies.
  - `chunk_size` 1200 with `chunk_overlap` 200 and `retrieval_k` 8 to improve recall across large codebases like 100K+ line monorepos.
