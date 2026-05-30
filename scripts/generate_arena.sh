#!/usr/bin/env bash
set -euo pipefail  # Exit on any error / undefined var / pipeline error

echo "🚀 Starting arena generation + grading..."

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ----------- OpenAI (GPT models) -----------
# Example: Generate with GPT-5.2
# uv run python -m memory_machines.arena.generate \
#     --model "gpt-4o" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 10

# uv run python -m memory_machines.arena.generate \
#     --model "gpt-4.1" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "o3" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "gpt-5.2" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "gpt-5.4" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "gpt-5.5" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# ----------- Anthropic (Claude models) -----------
# NOTE: For generation with Claude, specify base-url and auth-token
# The grading is ALWAYS done with Claude Sonnet 4.5 (hardcoded)
# uv run python -m memory_machines.arena.generate \
#     --model "claude-sonnet-4-5" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 18

# uv run python -m memory_machines.arena.generate \
#     --model "claude-3-7-sonnet-20250219" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "claude-opus-4-5-20251101" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "claude-opus-4-6" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "claude-3-haiku-20240307" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "claude-opus-4-7" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# uv run python -m memory_machines.arena.generate \
#     --model "claude-opus-4-8" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 4

# ----------- Google (Gemini models) -----------
# NOTE: Gemini supports reasoning-effort parameter
# uv run python -m memory_machines.arena.generate \
#     --model "gemini-2.5-pro" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --reasoning-effort "medium" \
#     --max-concurrency 18

# uv run python -m memory_machines.arena.generate \
#     --model "gemini-2.5-pro" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/trivia.txt" \
#     --reasoning-effort "medium" \
#     --max-concurrency 18

# uv run python -m memory_machines.arena.generate \
#     --model "gemini-3-pro-preview" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 18

# uv run python -m memory_machines.arena.generate \
#     --model "gemini-3.1-pro-preview" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 18

# uv run python -m memory_machines.arena.generate \
#     --model "gemini-3-flash-preview" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 18


# ----------- Open Source (via Cerebras) -----------
# uv run python -m memory_machines.arena.generate \
#     --model "gpt-oss-120b" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --instruction-file "memory_machines/arena/instruction/simple.txt" \
#     --max-concurrency 8

echo "✅ Done."
