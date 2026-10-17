"""``jarvis wake`` — wake word management command."""

from __future__ import annotations

import click
from rich.console import Console


@click.command(help="Gerenciamento do modo wake word (ativação por voz).")
@click.option(
    "--preparar",
    is_flag=True,
    default=False,
    help="Prepara e armazena em cache o áudio 'Sim, senhor.'",
)
def wake(preparar: bool) -> None:
    """Prepare and manage wake word activation assets."""
    console = Console(stderr=True)
    if preparar:
        from openjarvis.cli._voice_chat import VoiceSession
        from openjarvis.core.config import load_config
        from openjarvis.speech.wake import prepare_sim_senhor_cache

        config = load_config()
        session = VoiceSession(config)
        try:
            path, voice = prepare_sim_senhor_cache(session)
            console.print(f"[green]Cache preparado com sucesso:[/green] {path} (voz: {voice})")
        except Exception as exc:
            console.print(f"[yellow]Não foi possível gerar cache de áudio: {exc}[/yellow]")
            console.print("[dim]O assistente usará fallback em texto ao responder o chamado.[/dim]")
    else:
        console.print("Use 'jarvis wake --preparar' para gerar o cache de confirmação de ativação.")


__all__ = ["wake"]
