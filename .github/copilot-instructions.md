# GitHub Copilot instructions for scp_alg_test

**Purpose:** Help an AI coding agent (Copilot / automation) become productive quickly in this repository.

## Repository snapshot
- This repository currently contains a single file: `README.md`.
- No language-specific source files, tests, or CI workflows are present (as of this commit).

## Immediate goals for the agent
1. Confirm the repository owner's intent and target language(s) before making large changes.
2. Propose minimal, safe improvements (scaffolding, tests, CI) as small, reviewable PRs.
3. When uncertain, ask one focused question rather than guessing (examples below).

## Discovery checklist (what the agent should do first)
- Look for existing docs and agent rules: `**/{.github/copilot-instructions.md,AGENT.md,README.md}` (none found beyond `README.md`).
- Inspect the repository root for `src/`, `tests/`, `package.json`, `pyproject.toml`, `go.mod`, or other language indicators.
- If no indicators exist, **ask**: "Which language/tooling should I use to scaffold this project?" and wait for explicit instruction.

## Safe, high-value actions to propose or take
- Create a short proposal issue or draft PR describing the recommended scaffold (examples: `src/`, `tests/`, `README.md` updates).
- If owner confirms language = Python, propose adding:
  - `src/` for packages; `tests/` with pytest; a minimal `pyproject.toml` or GitHub Actions `ci.yml`.
  - Example test: `tests/test_placeholder.py` asserting a trivial import.
- If owner confirms language = JavaScript/TypeScript, propose adding `package.json`, `src/`, `test/`, and CI for `npm test`.
- When adding CI, keep workflows minimal and non-destructive (run linters/tests only).

## Pull request & commit guidance
- Make PRs small and focused; include a clear title and brief description.
- Each PR should include at least one test or a clear rationale for why a test isn’t needed yet.
- Use descriptive commit messages (e.g., "chore: add Python scaffold and basic test").

## What NOT to do without confirmation
- Do not add production deployment configs, secrets, or change external infra settings.
- Do not assume a test framework, linting rules, or package manager without explicit confirmation.

## Examples of clarifying questions (ask early)
- "Do you want this project to be Python, JavaScript/TypeScript, Go, or something else?"
- "Do you have a preferred test framework or CI provider/settings?"
- "Should I initialize the project with a minimal GitHub Actions CI that runs tests and lints?"

## Where to document agent decisions
- Add brief notes to the PR description summarizing the rationale for scaffolding choices and any assumptions made.
- Keep `.github/copilot-instructions.md` up to date if the repo structure or owner preferences change.

---

If you'd like, I can (1) open a small PR that scaffolds a language of your choice, or (2) start by creating an issue listing possible next steps—tell me which you prefer.