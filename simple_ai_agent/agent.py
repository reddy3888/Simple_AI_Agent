#!/usr/bin/env python3
"""
Project 1 — Simple AI Agent
============================
Architecture:
    User Input via CLI loop -> LLM Call -> Formatted Answer Output.

Features:
    - Official Google GenAI SDK (`google-genai`).
    - API key management with `python-dotenv`.
    - Default model: `gemini-2.5-flash`.
    - Configurable system prompts (persona presets or custom).
    - Flexible response formatting: Plain text or Structured JSON (via Pydantic).
    - Robust error handling: missing keys, network issues, and empty queries.
    - Interactive CLI loop with session commands (/format, /persona, /model, /help, /exit).
    - One-shot execution support (--query "..." flag).
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

# Suppress SDK informational warnings regarding Automatic Function Calling
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Graceful import of Google GenAI SDK
try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    print(
        "\n[ERROR] The 'google-genai' package is not installed.\n"
        "Please run:\n"
        "    pip install -r requirements.txt\n"
        "or:\n"
        "    pip install google-genai python-dotenv pydantic rich\n"
    )
    sys.exit(1)

# Graceful import of Rich for enhanced terminal display (with plain fallback)
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

# ============================================================================
# 1. Configuration & Default Settings
# ============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"

# Built-in persona presets for configurable system prompts
PERSONA_PRESETS: Dict[str, str] = {
    "assistant": (
        "You are a helpful, courteous, and knowledgeable AI assistant. "
        "Provide clear, accurate, and concise answers."
    ),
    "coder": (
        "You are an expert senior software engineer. Provide clean, well-commented, "
        "and idiomatic code snippets, accompanied by concise technical explanations "
        "and best practices."
    ),
    "tutor": (
        "You are a patient and encouraging Socratic tutor. Break complex concepts down "
        "step-by-step using intuitive analogies and relatable real-world examples."
    ),
    "analyst": (
        "You are a data-driven, objective analytical thinker. Deliver structured, "
        "rigorous breakdowns covering pros, cons, edge cases, and actionable takeaways."
    ),
    "creative": (
        "You are an imaginative and engaging creative writer. Use evocative vocabulary, "
        "vivid descriptions, and captivating narratives."
    ),
}


# ============================================================================
# 2. Structured Output Schema (Pydantic)
# ============================================================================

class StructuredAgentResponse(BaseModel):
    """Pydantic model defining the structured JSON schema for agent answers."""
    summary: str = Field(
        description="A concise 1-2 sentence executive overview answering the prompt."
    )
    category: str = Field(
        description="A 1-3 word high-level topic or domain classification."
    )
    key_points: list[str] = Field(
        description="A bulleted list of 2-5 core concepts, facts, or actionable steps."
    )
    detailed_answer: str = Field(
        description="The in-depth comprehensive answer to the user's inquiry."
    )
    confidence_or_notes: Optional[str] = Field(
        default=None,
        description="Any caveats, assumptions, or recommended next steps."
    )


# ============================================================================
# 3. Environment & API Key Verification
# ============================================================================

def init_environment() -> Optional[str]:
    """
    Loads environment variables from .env file and retrieves the Gemini API key.
    Checks both current working directory and the directory of this script.
    """
    # Attempt to load .env from current directory first, then script's directory
    load_dotenv()
    script_env_path = Path(__file__).resolve().parent / ".env"
    if script_env_path.exists():
        load_dotenv(dotenv_path=script_env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    return api_key.strip() if api_key else None


def display_missing_api_key_help():
    """Renders a helpful diagnostic guide when GEMINI_API_KEY is not configured."""
    msg = (
        "[bold red]Error: GEMINI_API_KEY is not configured.[/bold red]\n\n"
        "[bold cyan]How to set up your API Key:[/bold cyan]\n"
        "1. Obtain a free Gemini API key from Google AI Studio:\n"
        "   [underline]https://aistudio.google.com/app/apikey[/underline]\n\n"
        "2. Create a [bold].env[/bold] file inside the project directory:\n"
        "   - Copy the example: [italic]copy .env.example .env[/italic] (Windows)\n"
        "     or [italic]cp .env.example .env[/italic] (macOS/Linux)\n"
        "   - Open [bold].env[/bold] and set: [green]GEMINI_API_KEY=AIzaSy...[/green]\n\n"
        "3. Alternatively, export it in your terminal:\n"
        "   - PowerShell: [italic]$env:GEMINI_API_KEY=\"your_key\"[/italic]\n"
        "   - CMD:        [italic]set GEMINI_API_KEY=your_key[/italic]\n"
        "   - Bash/Zsh:   [italic]export GEMINI_API_KEY=\"your_key\"[/italic]\n"
    )
    if HAS_RICH:
        console.print(Panel(msg, title="[bold yellow]API Key Missing[/bold yellow]", border_style="red"))
    else:
        print("\n" + "=" * 60)
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Please configure your .env file or export GEMINI_API_KEY.")
        print("Get your API key at: https://aistudio.google.com/app/apikey")
        print("=" * 60 + "\n")


# ============================================================================
# 4. Core LLM Call & Error Handling
# ============================================================================

def call_gemini_agent(
    client: genai.Client,
    model: str,
    prompt: str,
    system_prompt: str,
    output_mode: str = "text",
) -> Tuple[bool, str]:
    """
    Executes a call to the Gemini API using the official google-genai SDK.

    Args:
        client: Initialized genai.Client instance.
        model: Name of the Gemini model (e.g. 'gemini-2.5-flash').
        prompt: User input prompt.
        system_prompt: Persona system instruction.
        output_mode: 'text' for standard formatting or 'json' for structured output.

    Returns:
        Tuple of (success: bool, content: str).
    """
    try:
        # Build generation configuration
        if output_mode == "json":
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=StructuredAgentResponse,
                temperature=0.3,  # Lower temperature for reliable structure
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )

        # Make the LLM API call
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        # Extract text content safely
        raw_text = response.text or ""
        return True, raw_text

    except errors.ClientError as e:
        # Handle 4xx HTTP client errors (auth, quota, bad request, model retirement)
        err_msg = str(e)
        if "403" in err_msg or "401" in err_msg or "API_KEY_INVALID" in err_msg:
            return False, (
                "Authentication failed: The supplied GEMINI_API_KEY appears invalid or unauthorized. "
                "Please verify your key at https://aistudio.google.com/app/apikey"
            )
        elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            return False, (
                "Rate limit or quota exceeded: You have reached the request limit. "
                "Please pause for a few seconds or check your quota."
            )
        elif "404" in err_msg or "no longer available" in err_msg.lower():
            # If the requested model (e.g. gemini-2.5-flash) is retired for this key, fallback
            fallback_model = "gemini-3.5-flash-lite"
            if model != fallback_model:
                try:
                    fb_response = client.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                        config=config,
                    )
                    return True, (fb_response.text or "")
                except Exception:
                    pass
            return False, (
                f"Model '{model}' is not available to this API key.\n"
                f"Try switching to an active model such as 'gemini-3.5-flash-lite' "
                f"by setting GEMINI_MODEL=gemini-3.5-flash-lite in your .env file or using --model gemini-3.5-flash-lite."
            )
        return False, f"Gemini Client Error: {err_msg}"

    except errors.ServerError as e:
        err_msg = str(e)
        if "503" in err_msg or "high demand" in err_msg.lower():
            fallback_model = "gemini-3.5-flash-lite"
            if model != fallback_model:
                try:
                    fb_response = client.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                        config=config,
                    )
                    return True, (fb_response.text or "")
                except Exception:
                    pass
        return False, f"Gemini Server Error (temporary upstream outage or high demand): {e}"

    except errors.APIError as e:
        return False, f"Gemini API Error: {e}"

    except Exception as e:
        # Handle network connectivity failures, timeouts, etc.
        err_name = e.__class__.__name__
        if "timeout" in err_name.lower() or "connect" in err_name.lower():
            return False, f"Network Connectivity Issue ({err_name}): Unable to reach Gemini API. Check your internet connection."
        return False, f"Unexpected Error ({err_name}): {e}"


# ============================================================================
# 5. Output Formatting & Display
# ============================================================================

def display_response(raw_content: str, output_mode: str, model_name: str):
    """Formats and prints the agent's response based on the chosen output mode."""
    if output_mode == "json":
        # Format as pretty JSON
        try:
            parsed = json.loads(raw_content)
            pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            pretty_json = raw_content

        if HAS_RICH:
            syntax = Syntax(pretty_json, "json", theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title=f"[bold green]Gemini Response [Structured JSON][/bold green] ({model_name})",
                subtitle="Format: application/json",
                border_style="green",
            )
            console.print(panel)
        else:
            print("\n" + "-" * 50)
            print(f"[Structured JSON Response - {model_name}]")
            print("-" * 50)
            print(pretty_json)
            print("-" * 50 + "\n")

    else:
        # Format as Markdown / Plain text
        if HAS_RICH:
            md = Markdown(raw_content)
            panel = Panel(
                md,
                title=f"[bold cyan]Gemini Response[/bold cyan] ({model_name})",
                border_style="cyan",
            )
            console.print(panel)
        else:
            print("\n" + "-" * 50)
            print(f"[Gemini Response - {model_name}]")
            print("-" * 50)
            print(raw_content)
            print("-" * 50 + "\n")


def display_error(error_message: str):
    """Renders formatted error notifications."""
    if HAS_RICH:
        console.print(
            Panel(
                f"[bold red]Request Failed:[/bold red]\n{error_message}",
                title="[bold yellow]Error Notice[/bold yellow]",
                border_style="red",
            )
        )
    else:
        print(f"\n[ERROR] {error_message}\n")


def display_banner(model: str, persona_name: str, output_mode: str):
    """Displays the startup header with current session parameters."""
    header = (
        "[bold cyan]============================================================[/bold cyan]\n"
        "           [bold white]Project 1: Simple AI Agent (Gemini)[/bold white]           \n"
        "[bold cyan]============================================================[/bold cyan]\n"
        f" [bold yellow]Model:[/bold yellow]   {model}\n"
        f" [bold yellow]Persona:[/bold yellow] {persona_name}\n"
        f" [bold yellow]Format:[/bold yellow]  {output_mode.upper()} ('text' or 'json')\n\n"
        "[dim]Type your prompt and press Enter.\n"
        "Special commands: [bold]/help[/bold], [bold]/format[/bold], [bold]/persona[/bold], [bold]/model[/bold], [bold]/exit[/bold][/dim]"
    )
    if HAS_RICH:
        console.print(Panel(header, border_style="blue"))
    else:
        print("=" * 60)
        print("           Project 1: Simple AI Agent (Gemini)            ")
        print("=" * 60)
        print(f" Model:   {model}")
        print(f" Persona: {persona_name}")
        print(f" Format:  {output_mode.upper()}")
        print(" Type your prompt and press Enter.")
        print(" Commands: /help, /format [text|json], /persona [name], /exit")
        print("=" * 60 + "\n")


def display_help():
    """Prints the help menu of in-session interactive commands."""
    if HAS_RICH:
        table = Table(title="Interactive Commands", border_style="yellow")
        table.add_column("Command", style="cyan bold", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_row("/help", "Display this help menu")
        table.add_row("/format [text|json]", "Switch between Plain Text and Structured JSON mode")
        table.add_row(
            "/persona [name|custom]",
            f"Change persona preset ({', '.join(PERSONA_PRESETS.keys())}) or provide custom instructions",
        )
        table.add_row("/model [name]", "Change Gemini model (e.g. gemini-2.5-flash, gemini-2.5-pro)")
        table.add_row("/clear", "Clear terminal screen")
        table.add_row("/exit or /quit", "Exit the agent session")
        console.print(table)
    else:
        print("\nCommands:")
        print("  /help                 - Show this help menu")
        print("  /format [text|json]   - Switch output mode")
        print(f"  /persona [name]       - Change persona ({', '.join(PERSONA_PRESETS.keys())})")
        print("  /model [name]         - Switch Gemini model")
        print("  /clear                - Clear the screen")
        print("  /exit, /quit          - Exit application\n")


# ============================================================================
# 6. Interactive CLI Loop
# ============================================================================

def run_cli_loop(
    client: genai.Client,
    model: str,
    persona_name: str,
    system_prompt: str,
    output_mode: str,
):
    """
    Runs the interactive Read-Eval-Print loop for the agent.
    Handles user queries, dynamic commands, and graceful interrupts.
    """
    current_model = model
    current_persona_name = persona_name
    current_system_prompt = system_prompt
    current_format = output_mode

    display_banner(current_model, current_persona_name, current_format)

    while True:
        try:
            # Prompt user for input
            prompt_symbol = f"[{current_format.upper()}] You > "
            user_input = input(prompt_symbol).strip()

            # Handle empty input
            if not user_input:
                if HAS_RICH:
                    console.print("[dim yellow]Input is empty. Please enter a prompt or type /help.[/dim yellow]")
                else:
                    print("Input is empty. Please enter a prompt or type /help.")
                continue

            # Command handling: /exit or /quit
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                if HAS_RICH:
                    console.print("[bold green]Thank you for using Simple AI Agent. Goodbye![/bold green]")
                else:
                    print("Thank you for using Simple AI Agent. Goodbye!")
                break

            # Command handling: /help
            if user_input.lower() == "/help":
                display_help()
                continue

            # Command handling: /clear
            if user_input.lower() == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                display_banner(current_model, current_persona_name, current_format)
                continue

            # Command handling: /format
            if user_input.lower().startswith("/format"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].lower() in ("text", "json"):
                    current_format = parts[1].lower()
                    if HAS_RICH:
                        console.print(f"[bold green]Output format set to: {current_format.upper()}[/bold green]")
                    else:
                        print(f"Output format set to: {current_format.upper()}")
                else:
                    target = "json" if current_format == "text" else "text"
                    current_format = target
                    if HAS_RICH:
                        console.print(f"[bold green]Toggled output format to: {current_format.upper()}[/bold green]")
                    else:
                        print(f"Toggled output format to: {current_format.upper()}")
                continue

            # Command handling: /model
            if user_input.lower().startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    current_model = parts[1].strip()
                    if HAS_RICH:
                        console.print(f"[bold green]Active model changed to: {current_model}[/bold green]")
                    else:
                        print(f"Active model changed to: {current_model}")
                else:
                    if HAS_RICH:
                        console.print(f"[dim]Current model is: {current_model}. Specify a model name, e.g., /model gemini-2.5-flash[/dim]")
                    else:
                        print(f"Current model is: {current_model}")
                continue

            # Command handling: /persona
            if user_input.lower().startswith("/persona"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    key = parts[1].strip().lower()
                    if key in PERSONA_PRESETS:
                        current_persona_name = key
                        current_system_prompt = PERSONA_PRESETS[key]
                        if HAS_RICH:
                            console.print(f"[bold green]Persona switched to '{key}': {current_system_prompt}[/bold green]")
                        else:
                            print(f"Persona switched to '{key}': {current_system_prompt}")
                    else:
                        # User specified a custom system instruction
                        current_persona_name = "custom"
                        current_system_prompt = parts[1].strip()
                        if HAS_RICH:
                            console.print(f"[bold green]Custom persona instruction set:[/bold green] {current_system_prompt}")
                        else:
                            print(f"Custom persona set: {current_system_prompt}")
                else:
                    presets_list = ", ".join(PERSONA_PRESETS.keys())
                    if HAS_RICH:
                        console.print(f"[dim]Current persona: {current_persona_name}. Available presets: {presets_list}[/dim]")
                    else:
                        print(f"Current persona: {current_persona_name}. Available presets: {presets_list}")
                continue

            # Regular prompt: Call LLM
            if HAS_RICH:
                with console.status(f"[bold cyan]Thinking ({current_model})...[/bold cyan]", spinner="dots"):
                    success, result = call_gemini_agent(
                        client=client,
                        model=current_model,
                        prompt=user_input,
                        system_prompt=current_system_prompt,
                        output_mode=current_format,
                    )
            else:
                print(f"Thinking ({current_model})...")
                success, result = call_gemini_agent(
                    client=client,
                    model=current_model,
                    prompt=user_input,
                    system_prompt=current_system_prompt,
                    output_mode=current_format,
                )

            # Display response or error
            if success:
                display_response(result, current_format, current_model)
            else:
                display_error(result)

        except KeyboardInterrupt:
            # Graceful Ctrl+C handling
            print("\n")
            if HAS_RICH:
                console.print("[bold yellow]Session interrupted by user (Ctrl+C). Goodbye![/bold yellow]")
            else:
                print("Session interrupted by user (Ctrl+C). Goodbye!")
            break
        except EOFError:
            # Graceful Ctrl+D / pipe termination handling
            print("\n")
            break


# ============================================================================
# 7. CLI Argument Parsing & Entry Point
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Simple AI Agent."""
    parser = argparse.ArgumentParser(
        description="Simple AI Agent - Project 1 (Google Gemini SDK)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py\n"
            "  python agent.py --json\n"
            "  python agent.py --persona coder\n"
            "  python agent.py --system-prompt 'You are an astronomy professor.'\n"
            "  python agent.py --query 'Explain quantum computing in one sentence.' --json\n"
        ),
    )

    parser.add_argument(
        "-m",
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        help=f"Gemini model ID to use (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "-p",
        "--persona",
        default="assistant",
        choices=list(PERSONA_PRESETS.keys()) + ["custom"],
        help="Predefined persona to adopt (default: assistant)",
    )

    parser.add_argument(
        "-s",
        "--system-prompt",
        default=None,
        help="Custom system prompt (overrides predefined persona)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json"],
        default="text",
        help="Response format: 'text' or 'json' (default: text)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Convenience shortcut to set output format to structured JSON",
    )

    parser.add_argument(
        "-q",
        "--query",
        default=None,
        help="Run a single one-shot query and exit immediately instead of entering the CLI loop",
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit Gemini API key (defaults to GEMINI_API_KEY environment variable)",
    )

    return parser.parse_args()


def main():
    """Main application entry point."""
    # 1. Load environment variables from .env
    env_api_key = init_environment()
    args = parse_arguments()

    # Resolve API key
    api_key = args.api_key or env_api_key
    if not api_key:
        display_missing_api_key_help()
        sys.exit(1)

    # 2. Resolve Persona & System Prompt
    if args.system_prompt:
        persona_name = "custom"
        system_prompt = args.system_prompt
    else:
        persona_name = args.persona
        system_prompt = PERSONA_PRESETS.get(persona_name, PERSONA_PRESETS["assistant"])

    # 3. Resolve Output Format
    output_mode = "json" if args.json else args.format

    # 4. Initialize Google GenAI Client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        display_error(f"Failed to initialize Google GenAI Client: {e}")
        sys.exit(1)

    # 5. One-shot mode or interactive loop
    if args.query:
        # Check for empty query
        clean_query = args.query.strip()
        if not clean_query:
            display_error("The provided query is empty.")
            sys.exit(1)

        success, result = call_gemini_agent(
            client=client,
            model=args.model,
            prompt=clean_query,
            system_prompt=system_prompt,
            output_mode=output_mode,
        )
        if success:
            display_response(result, output_mode, args.model)
            sys.exit(0)
        else:
            display_error(result)
            sys.exit(1)
    else:
        # Interactive CLI Loop
        run_cli_loop(
            client=client,
            model=args.model,
            persona_name=persona_name,
            system_prompt=system_prompt,
            output_mode=output_mode,
        )


if __name__ == "__main__":
    main()
