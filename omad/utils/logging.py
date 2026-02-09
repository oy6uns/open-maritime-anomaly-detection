"""Logging utilities using Rich library for beautiful terminal output."""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Global console instance
console = Console()


def create_progress() -> Progress:
    """
    Create a Rich progress bar for batch operations.

    Returns:
        Progress instance with configured columns
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def create_summary_table(title: str, data: dict, title_style: str = "bold cyan") -> Table:
    """
    Create a formatted summary table.

    Args:
        title: Table title
        data: Dictionary of metrics to display
        title_style: Style for the title (default: "bold cyan")

    Returns:
        Rich Table instance
    """
    table = Table(title=title, title_style=title_style, show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    for key, value in data.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = f"{value:,}" if isinstance(value, int) else str(value)
        table.add_row(formatted_key, formatted_value)

    return table


def log_stage_header(stage: int, description: str):
    """
    Display stage header banner.

    Args:
        stage: Stage number (1-4)
        description: Stage description
    """
    console.print(Panel(
        f"[bold white]Stage {stage}:[/bold white] {description}",
        border_style="blue",
        padding=(1, 2),
    ))


def log_config(config_dict: dict):
    """
    Display configuration parameters.

    Args:
        config_dict: Dictionary of configuration key-value pairs
    """
    console.print("\n[bold]Configuration:[/bold]")
    for key, value in config_dict.items():
        console.print(f"  [cyan]{key}:[/cyan] {value}")
    console.print()


def log_success(message: str):
    """
    Display success message.

    Args:
        message: Success message
    """
    console.print(f"[green]✓[/green] {message}")


def log_error(message: str):
    """
    Display error message.

    Args:
        message: Error message
    """
    console.print(f"[red]✗[/red] {message}")


def log_warning(message: str):
    """
    Display warning message.

    Args:
        message: Warning message
    """
    console.print(f"[yellow]⚠[/yellow] {message}")


def log_info(message: str):
    """
    Display info message.

    Args:
        message: Info message
    """
    console.print(f"[blue]ℹ[/blue] {message}")
