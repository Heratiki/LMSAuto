# LMSAuto Command Line Interface
# Entry point for the application

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.tree import Tree
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.columns import Columns
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.console import Group
from rich import box
import threading
import time
import sys
import select
import termios
import tty
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from lmsauto.shared.context import SharedContext
from lmsauto.profiler import SystemProfiler
from lmsauto.scanner import ModelScanner, LMStudioScanner
from lmsauto.optimizer.prompt_optimizer import PromptWizardOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

@dataclass
class AppState:
    """Application state for the TUI."""
    selected_menu_item: int = 0
    models: Optional[Dict[str, Any]] = None
    hardware_specs: Optional[Dict[str, str]] = None
    optimization_log: List[str] = None
    is_running: bool = True
    
    def __post_init__(self):
        if self.optimization_log is None:
            self.optimization_log = []

class LMSAutoTUI:
    """LMSAuto Terminal User Interface using Rich Layout."""
    
    def __init__(self, context: SharedContext):
        self.context = context
        self.console = Console()
        self.state = AppState()
        self.layout = Layout()
        self.menu_items = [
            "🔍 Scan for Models",
            "🖥️  Profile System Hardware", 
            "🧠 Optimize Prompts",
            "⚙️  Full Workflow",
            "❓ Help",
            "🚪 Exit"
        ]
        self.setup_layout()
    
    def setup_layout(self):
        """Setup the main layout structure."""
        # Get terminal size
        size = self.console.size
        
        # Create main layout splits - adjust based on terminal height
        if size.height < 30:
            # Smaller terminal - simpler layout
            self.layout.split_column(
                Layout(name="header", size=1),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=1)
            )
            
            # Just split into menu and output for small terminals
            self.layout["main"].split_row(
                Layout(name="menu", ratio=1),
                Layout(name="output", ratio=2)
            )
        else:
            # Normal terminal - full layout
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="footer", size=2)
            )
            
            # Split main area into top and bottom
            self.layout["main"].split_column(
                Layout(name="top", minimum_size=15, ratio=2),
                Layout(name="bottom", minimum_size=10, ratio=3)
            )
            
            # Split top area into left and right
            self.layout["top"].split_row(
                Layout(name="menu", minimum_size=25, ratio=1),
                Layout(name="info", ratio=2)
            )
            
            # Split info area into models and hardware
            self.layout["info"].split_column(
                Layout(name="models", ratio=1),
                Layout(name="hardware", ratio=1)
            )
    
    def get_key(self):
        """Get a single keypress from stdin (Linux/macOS)."""
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC sequence
                    ch += sys.stdin.read(2)
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            # Fallback for non-Unix systems
            return input()
    
    def create_menu_panel(self) -> Panel:
        """Create the menu panel with current selection highlighted."""
        menu_lines = []
        for i, item in enumerate(self.menu_items):
            # Shorten menu items for better fit
            short_item = item.replace("🖥️  Profile System Hardware", "🖥️ Hardware")
            short_item = short_item.replace("⚙️  Full Workflow", "⚙️ Workflow")
            
            if i == self.state.selected_menu_item:
                menu_lines.append(f"[black on cyan]► {i+1}. {short_item}[/black on cyan]")
            else:
                menu_lines.append(f"  {i+1}. {short_item}")
        
        menu_content = "\n".join(menu_lines)
        menu_content += "\n\n[dim]↑↓ or numbers\nEnter=select, q=quit[/dim]"
        
        return Panel(
            menu_content,
            title="[bold blue]Menu[/bold blue]",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def create_models_panel(self) -> Panel:
        """Create the models panel."""
        if not self.state.models:
            content = "[dim]No models scanned yet.\nSelect 'Scan for Models' to discover available models.[/dim]"
        else:
            table = Table(show_header=True, box=None, show_lines=False)
            table.add_column("Model", style="cyan", max_width=30)
            table.add_column("Platform", style="green", max_width=10)
            table.add_column("Source", style="yellow", max_width=8)
            
            for model_key, model_info in list(self.state.models.items())[:10]:  # Show first 10
                source = model_info.metadata.get('source', 'unknown') if hasattr(model_info, 'metadata') and model_info.metadata else 'unknown'
                table.add_row(
                    model_info.name[:28] + "..." if len(model_info.name) > 28 else model_info.name,
                    model_info.platform,
                    source
                )
            
            if len(self.state.models) > 10:
                content = Group(table, f"\n[dim]... and {len(self.state.models) - 10} more models[/dim]")
            else:
                content = table
        
        return Panel(
            content,
            title=f"[bold green]Models ({len(self.state.models) if self.state.models else 0})[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )
    
    def create_hardware_panel(self) -> Panel:
        """Create the hardware panel."""
        if not self.state.hardware_specs:
            content = "[dim]No hardware profile available.\nSelect 'Profile System Hardware' to analyze your system.[/dim]"
        else:
            table = Table(show_header=False, box=None, show_lines=False)
            table.add_column("Component", style="cyan", max_width=12)
            table.add_column("Value", style="magenta", max_width=20)
            
            # Show key specs
            key_specs = ['cpu_physical_cores', 'cpu_total_cores', 'ram_total_gb', 'ram_available_gb', 'gpu_available']
            for key in key_specs:
                if key in self.state.hardware_specs:
                    component = key.replace('_', ' ').title()
                    value = str(self.state.hardware_specs[key])
                    if len(value) > 18:
                        value = value[:15] + "..."
                    table.add_row(component, value)
            
            content = table
        
        return Panel(
            content,
            title="[bold magenta]Hardware Profile[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED
        )
    
    def create_output_panel(self) -> Panel:
        """Create the output panel for optimization logs."""
        if not self.state.optimization_log:
            content = "[dim]Optimization output will appear here.\nSelect 'Optimize Prompts' to start.[/dim]"
        else:
            # Show last 15 lines
            recent_logs = self.state.optimization_log[-15:]
            content = "\n".join(recent_logs)
        
        return Panel(
            content,
            title="[bold yellow]Optimization Output[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def create_header(self) -> Panel:
        """Create the header panel."""
        return Panel(
            "[bold blue]LMSAuto[/bold blue] - LM Studio Autonomous Model Settings Optimizer",
            style="bold cyan"
        )
    
    def create_footer(self) -> Panel:
        """Create the footer panel."""
        return Panel(
            "[dim]Press 'q' to quit, 'r' to refresh, or use menu navigation[/dim]",
            style="dim"
        )
    
    def update_layout(self):
        """Update all layout components."""
        size = self.console.size
        
        self.layout["header"].update(self.create_header())
        self.layout["menu"].update(self.create_menu_panel())
        
        if size.height < 30:
            # Simple layout for small terminals
            self.layout["output"].update(self.create_output_panel())
        else:
            # Full layout for normal terminals
            self.layout["models"].update(self.create_models_panel())
            self.layout["hardware"].update(self.create_hardware_panel())
            self.layout["bottom"].update(self.create_output_panel())
        
        self.layout["footer"].update(self.create_footer())
    
    def log_output(self, message: str):
        """Add a message to the optimization log."""
        timestamp = time.strftime("%H:%M:%S")
        self.state.optimization_log.append(f"[{timestamp}] {message}")
        # Keep only last 50 entries
        if len(self.state.optimization_log) > 50:
            self.state.optimization_log = self.state.optimization_log[-50:]
    
    def scan_models(self):
        """Scan for models and update state."""
        self.log_output("[cyan]Starting model scan...[/cyan]")
        try:
            scanner = ModelScanner(self.context)
            lm_studio_scanner = LMStudioScanner()
            scanner.register_scanner(lm_studio_scanner)
            scanner.scan()
            
            self.state.models = self.context.get_all_models()
            if self.state.models:
                self.log_output(f"[green]✅ Found {len(self.state.models)} models[/green]")
            else:
                self.log_output("[red]❌ No models found. Check LM Studio connection.[/red]")
        except Exception as e:
            self.log_output(f"[red]❌ Scan failed: {str(e)}[/red]")
    
    def profile_hardware(self):
        """Profile system hardware and update state."""
        self.log_output("[cyan]Starting hardware profiling...[/cyan]")
        try:
            profiler = SystemProfiler(self.context)
            profiler.profile_system()
            
            self.state.hardware_specs = self.context.get_hardware_specs()
            if self.state.hardware_specs:
                self.log_output("[green]✅ Hardware profiling complete[/green]")
            else:
                self.log_output("[red]❌ Hardware profiling failed[/red]")
        except Exception as e:
            self.log_output(f"[red]❌ Profiling failed: {str(e)}[/red]")
    
    def optimize_prompts(self):
        """Optimize prompts for selected models."""
        if not self.state.models:
            self.log_output("[red]❌ No models available. Scan for models first.[/red]")
            return
        
        self.log_output("[cyan]Starting prompt optimization...[/cyan]")
        
        # Use first 3 models for optimization
        selected_models = dict(list(self.state.models.items())[:3])
        self.log_output(f"[yellow]Optimizing {len(selected_models)} models[/yellow]")
        
        try:
            optimizer = PromptWizardOptimizer(self.context)
            
            for model_key, model_info in selected_models.items():
                self.log_output(f"[blue]🔧 Optimizing: {model_info.name}[/blue]")
                
                try:
                    result = optimizer.optimize(
                        model_key=model_key,
                        task_description="Generate helpful and accurate responses",
                        task_type="instruction",
                        max_iterations=1  # Reduced for faster demo
                    )
                    
                    improvement = result.improvement
                    effectiveness = result.metrics.get('overall_effectiveness', 0)
                    self.log_output(f"[green]✅ {model_info.name}: +{improvement:.3f} effectiveness: {effectiveness:.3f}[/green]")
                    
                except Exception as e:
                    self.log_output(f"[red]❌ {model_info.name}: Failed ({str(e)[:30]}...)[/red]")
            
            self.log_output("[green]🎉 Optimization complete![/green]")
            
        except Exception as e:
            self.log_output(f"[red]❌ Optimization failed: {str(e)}[/red]")
    
    def run_full_workflow(self):
        """Run the complete workflow."""
        self.log_output("[cyan]🚀 Starting full workflow...[/cyan]")
        self.scan_models()
        time.sleep(0.5)  # Brief pause for visual effect
        self.profile_hardware()
        time.sleep(0.5)
        self.optimize_prompts()
        self.log_output("[green]✨ Full workflow complete![/green]")
    
    def show_help(self):
        """Show help information."""
        help_lines = [
            "[bold cyan]LMSAuto Help[/bold cyan]",
            "",
            "[yellow]Navigation:[/yellow]",
            "• Use ↑↓ arrow keys or numbers 1-6",
            "• Press Enter to select menu item",
            "• Press 'q' to quit, 'r' to refresh",
            "",
            "[yellow]Features:[/yellow]",
            "• Model discovery via LM Studio API",
            "• Hardware profiling and analysis", 
            "• AI-powered prompt optimization",
            "• Real-time status updates"
        ]
        
        for line in help_lines:
            self.log_output(line)
    
    def handle_menu_selection(self):
        """Handle the current menu selection."""
        selection = self.state.selected_menu_item
        
        if selection == 0:  # Scan for Models
            threading.Thread(target=self.scan_models, daemon=True).start()
        elif selection == 1:  # Profile System Hardware
            threading.Thread(target=self.profile_hardware, daemon=True).start()
        elif selection == 2:  # Optimize Prompts
            threading.Thread(target=self.optimize_prompts, daemon=True).start()
        elif selection == 3:  # Full Workflow
            threading.Thread(target=self.run_full_workflow, daemon=True).start()
        elif selection == 4:  # Help
            self.show_help()
        elif selection == 5:  # Exit
            self.state.is_running = False
    
    def run(self):
        """Run the TUI application."""
        # Check terminal size
        size = self.console.size
        if size.height < 20 or size.width < 60:
            self.console.print(f"[red]Terminal too small: {size.width}x{size.height}[/red]")
            self.console.print("[yellow]Minimum required: 60x20[/yellow]")
            self.console.print("[cyan]Please resize your terminal and try again.[/cyan]")
            return
        
        self.log_output(f"[green]Welcome to LMSAuto! Terminal: {size.width}x{size.height}[/green]")
        
        try:
            with Live(self.layout, console=self.console, screen=True, redirect_stderr=False) as live:
                while self.state.is_running:
                    try:
                        self.update_layout()
                        live.update(self.layout)
                        
                        # Simple input handling - just get one character
                        key = self.get_key()
                        
                        if key == 'q' or key == 'Q':
                            self.state.is_running = False
                        elif key == 'r' or key == 'R':
                            self.log_output("[blue]🔄 Refreshing display...[/blue]")
                        elif key == '\x1b[A':  # Up arrow
                            self.state.selected_menu_item = (self.state.selected_menu_item - 1) % len(self.menu_items)
                        elif key == '\x1b[B':  # Down arrow
                            self.state.selected_menu_item = (self.state.selected_menu_item + 1) % len(self.menu_items)
                        elif key == '\r' or key == '\n':  # Enter
                            self.handle_menu_selection()
                        elif key.isdigit():
                            num = int(key) - 1
                            if 0 <= num < len(self.menu_items):
                                self.state.selected_menu_item = num
                                self.handle_menu_selection()
                        
                    except KeyboardInterrupt:
                        self.state.is_running = False
                    except Exception as e:
                        self.log_output(f"[red]Error: {str(e)}[/red]")
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        except Exception as e:
            self.console.print(f"\n[red]TUI Error: {e}[/red]")
        
        self.console.print("\n[green]Thank you for using LMSAuto! 👋[/green]")

def show_main_menu():
    """Display the main menu options."""
    menu_options = [
        "[1] 🔍 Scan for Models",
        "[2] 🖥️  Profile System Hardware", 
        "[3] 🧠 Optimize Prompts",
        "[4] ⚙️  Full Workflow (Scan + Profile + Optimize)",
        "[5] ❓ Help",
        "[6] 🚪 Exit"
    ]
    
    menu_panel = Panel(
        "\n".join(menu_options),
        title="[bold blue]LMSAuto Main Menu[/bold blue]",
        style="cyan",
        box=box.ROUNDED
    )
    
    console.print()
    console.print(Align.center(menu_panel))
    console.print()

def get_optimization_settings():
    """Get optimization settings from user."""
    console.print("[bold yellow]Prompt Optimization Settings[/bold yellow]")
    
    task_types = ["instruction", "chat", "completion"]
    console.print("Task Types: " + " | ".join(f"[cyan]{i+1}[/cyan]. {t}" for i, t in enumerate(task_types)))
    
    while True:
        choice = IntPrompt.ask("Select task type", default=1)
        if 1 <= choice <= len(task_types):
            task_type = task_types[choice - 1]
            break
        console.print("[red]Invalid choice. Please select 1, 2, or 3.[/red]")
    
    task_description = Prompt.ask(
        "Enter task description", 
        default="Generate helpful and accurate responses"
    )
    
    max_iterations = IntPrompt.ask(
        "Maximum optimization iterations", 
        default=2,
        choices=["1", "2", "3"]
    )
    
    return task_type, task_description, max_iterations

def select_models_for_optimization(models):
    """Allow user to select specific models for optimization."""
    if not models:
        return {}
    
    console.print(f"\n[bold cyan]Select Models for Optimization ({len(models)} available)[/bold cyan]")
    
    # Display models in a table with numbers
    table = Table(title="Available Models", box=box.ROUNDED)
    table.add_column("ID", style="yellow", width=4)
    table.add_column("Model Name", style="cyan")
    table.add_column("Platform", style="green")
    
    model_list = list(models.items())
    for i, (model_key, model_info) in enumerate(model_list, 1):
        table.add_row(str(i), model_info.name, model_info.platform)
    
    console.print(table)
    
    # Get user selection
    console.print("\nSelection options:")
    console.print("[cyan]• Enter specific numbers (e.g., 1,3,5)[/cyan]")
    console.print("[cyan]• Enter 'all' for all models[/cyan]")
    console.print("[cyan]• Enter 'first3' for first 3 models[/cyan]")
    console.print("[cyan]• Press Enter for first 3 models (default)[/cyan]")
    
    selection = Prompt.ask("Select models", default="first3").strip().lower()
    
    if selection == "all":
        return models
    elif selection == "first3" or selection == "":
        return dict(list(models.items())[:3])
    else:
        try:
            # Parse comma-separated numbers
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_models = {}
            for idx in indices:
                if 0 <= idx < len(model_list):
                    key, model = model_list[idx]
                    selected_models[key] = model
            return selected_models
        except ValueError:
            console.print("[red]Invalid selection. Using first 3 models.[/red]")
            return dict(list(models.items())[:3])

def show_help():
    """Display help information."""
    help_content = """
[bold cyan]LMSAuto - LM Studio Autonomous Model Settings Optimizer[/bold cyan]

[bold yellow]Features:[/bold yellow]
• [green]Model Discovery[/green]: Automatically finds models via LM Studio API
• [green]System Profiling[/green]: Analyzes your hardware capabilities  
• [green]Prompt Optimization[/green]: Uses PromptWizard algorithm to improve prompts
• [green]Rich Terminal UI[/green]: Professional, colorful interface

[bold yellow]Workflow:[/bold yellow]
1. Connect to LM Studio (ensure it's running on localhost:1234)
2. Scan for available models 
3. Profile system hardware (optional)
4. Select models and optimize prompts using AI-driven refinement

[bold yellow]Optimization Process:[/bold yellow]
• Generates multiple prompt variants
• Evaluates each on clarity, task-specificity, structure, guidance
• Iteratively refines using self-critique
• Returns best performing optimized prompt

[bold yellow]Requirements:[/bold yellow]
• LM Studio running locally
• Models loaded in LM Studio
• Python 3.11+ with dependencies installed
    """
    
    console.print(Panel(
        help_content,
        title="Help & Information",
        style="blue",
        box=box.ROUNDED
    ))

def interactive_mode(context):
    """Run LMSAuto in interactive menu mode."""
    # Welcome banner
    console.print(Panel.fit(
        "[bold blue]LMSAuto[/bold blue] - LM Studio Autonomous Model Settings Optimizer\n"
        "[dim]Interactive Mode[/dim]",
        style="bold cyan"
    ))
    
    models = None
    
    while True:
        show_main_menu()
        
        try:
            choice = IntPrompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == 1:  # Scan for Models
                with console.status("[bold green]Scanning for models..."):
                    scanner = ModelScanner(context)
                    lm_studio_scanner = LMStudioScanner()
                    scanner.register_scanner(lm_studio_scanner)
                    scanner.scan()
                
                models = context.get_all_models()
                if models:
                    table = Table(title=f"Discovered Models ({len(models)})", box=box.ROUNDED)
                    table.add_column("Model Name", style="cyan")
                    table.add_column("Platform", style="green")
                    table.add_column("Source", style="yellow")
                    
                    for model_key, model_info in models.items():
                        source = model_info.metadata.get('source', 'unknown') if hasattr(model_info, 'metadata') and model_info.metadata else 'unknown'
                        table.add_row(model_info.name, model_info.platform, source)
                    
                    console.print(table)
                    console.print(f"[green]✅ Found {len(models)} models[/green]")
                else:
                    console.print(Panel(
                        "[bold red]No models found[/bold red]\n"
                        "Make sure LM Studio is running with models loaded.",
                        title="Warning",
                        style="yellow"
                    ))
                
                Prompt.ask("\nPress Enter to continue")
                
            elif choice == 2:  # Profile System Hardware
                with console.status("[bold green]Profiling system hardware..."):
                    profiler = SystemProfiler(context)
                    profiler.profile_system()
                
                specs = context.get_hardware_specs()
                if specs:
                    table = Table(title="System Hardware Specifications", box=box.ROUNDED)
                    table.add_column("Component", style="cyan", no_wrap=True)
                    table.add_column("Value", style="magenta")
                    
                    for key, value in specs.items():
                        component = key.replace('_', ' ').title()
                        table.add_row(component, str(value))
                    
                    console.print(table)
                    console.print("[green]✅ System profiling complete[/green]")
                
                Prompt.ask("\nPress Enter to continue")
                
            elif choice == 3:  # Optimize Prompts
                if not models:
                    if Confirm.ask("No models scanned yet. Scan for models first?"):
                        # Scan for models first
                        with console.status("[bold green]Scanning for models..."):
                            scanner = ModelScanner(context)
                            lm_studio_scanner = LMStudioScanner()
                            scanner.register_scanner(lm_studio_scanner)
                            scanner.scan()
                        models = context.get_all_models()
                
                if models:
                    # Get optimization settings
                    task_type, task_description, max_iterations = get_optimization_settings()
                    
                    # Select models
                    selected_models = select_models_for_optimization(models)
                    
                    if selected_models:
                        console.print(f"\n[bold blue]Starting Optimization for {len(selected_models)} models[/bold blue]")
                        
                        # Run optimization
                        optimizer = PromptWizardOptimizer(context)
                        results = []
                        
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(),
                            TaskProgressColumn(),
                            console=console
                        ) as progress:
                            task = progress.add_task("[green]Optimizing prompts...", total=len(selected_models))
                            
                            for model_key, model_info in selected_models.items():
                                progress.update(task, description=f"[green]Optimizing: {model_info.name}")
                                
                                try:
                                    result = optimizer.optimize(
                                        model_key=model_key,
                                        task_description=task_description,
                                        task_type=task_type,
                                        max_iterations=max_iterations
                                    )
                                    results.append(result)
                                    console.print(f"[green]✅[/green] {model_info.name}: Improvement {result.improvement:.3f}")
                                except Exception as e:
                                    console.print(f"[red]❌[/red] {model_info.name}: Optimization failed")
                                
                                progress.advance(task)
                        
                        # Show results
                        if results:
                            summary_table = Table(title="Optimization Results", box=box.ROUNDED)
                            summary_table.add_column("Model", style="cyan")
                            summary_table.add_column("Improvement", style="green")
                            summary_table.add_column("Variants Tested", style="yellow")
                            summary_table.add_column("Effectiveness", style="magenta")
                            
                            for result in results:
                                summary_table.add_row(
                                    result.model_key.replace('lm_studio_', ''),
                                    f"{result.improvement:.3f}",
                                    str(result.variants_tested),
                                    f"{result.metrics.get('overall_effectiveness', 0):.3f}"
                                )
                            
                            console.print(summary_table)
                            
                            # Show best result
                            best_result = max(results, key=lambda r: r.metrics.get('overall_effectiveness', 0))
                            console.print(Panel(
                                best_result.optimized_prompt,
                                title=f"Best Optimized Prompt - {best_result.model_key.replace('lm_studio_', '')}",
                                style="cyan"
                            ))
                    else:
                        console.print("[red]No models selected.[/red]")
                else:
                    console.print("[red]No models available. Please scan for models first.[/red]")
                
                Prompt.ask("\nPress Enter to continue")
                
            elif choice == 4:  # Full Workflow
                console.print("[bold blue]Running Full Workflow...[/bold blue]")
                
                # 1. Scan models
                with console.status("[bold green]Scanning for models..."):
                    scanner = ModelScanner(context)
                    lm_studio_scanner = LMStudioScanner()
                    scanner.register_scanner(lm_studio_scanner)
                    scanner.scan()
                models = context.get_all_models()
                
                if not models:
                    console.print("[red]No models found. Cannot proceed with workflow.[/red]")
                    Prompt.ask("\nPress Enter to continue")
                    continue
                
                # 2. Profile system
                with console.status("[bold green]Profiling system..."):
                    profiler = SystemProfiler(context)
                    profiler.profile_system()
                
                # 3. Quick optimization settings
                console.print("[yellow]Using default optimization settings:[/yellow]")
                console.print("• Task Type: instruction")
                console.print("• Task: Generate helpful responses")
                console.print("• Iterations: 2")
                console.print("• Models: First 3")
                
                if Confirm.ask("Proceed with these settings?"):
                    selected_models = dict(list(models.items())[:3])
                    optimizer = PromptWizardOptimizer(context)
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=console
                    ) as progress:
                        task = progress.add_task("[green]Full workflow...", total=len(selected_models))
                        
                        for model_key, model_info in selected_models.items():
                            progress.update(task, description=f"[green]Optimizing: {model_info.name}")
                            try:
                                result = optimizer.optimize(
                                    model_key=model_key,
                                    task_description="Generate helpful and accurate responses",
                                    task_type="instruction",
                                    max_iterations=2
                                )
                                console.print(f"[green]✅[/green] {model_info.name} optimized")
                            except Exception:
                                console.print(f"[red]❌[/red] {model_info.name} failed")
                            progress.advance(task)
                    
                    console.print("[green]✅ Full workflow complete![/green]")
                
                Prompt.ask("\nPress Enter to continue")
                
            elif choice == 5:  # Help
                show_help()
                Prompt.ask("\nPress Enter to continue")
                
            elif choice == 6:  # Exit
                console.print("[green]Thank you for using LMSAuto! 👋[/green]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
        except Exception as e:
            console.print(Panel(
                f"[bold red]Error: {e}[/bold red]",
                title="Error",
                style="red"
            ))

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Update root logger level
    logging.getLogger().setLevel(numeric_level)
    logger.setLevel(numeric_level)

def main():
    """Main function for the LMSAuto CLI."""
    parser = argparse.ArgumentParser(
        description="LMSAuto - LM Studio Autonomous Model Settings Optimizer"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive menu mode (default if no other options provided)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan for models without generating profiles"
    )
    
    parser.add_argument(
        "--profile-system",
        action="store_true",
        help="Profile system hardware specs"
    )
    
    parser.add_argument(
        "--optimize-prompts",
        action="store_true",
        help="Optimize prompts for discovered models"
    )
    
    parser.add_argument(
        "--task-type",
        choices=['completion', 'chat', 'instruction'],
        default='instruction',
        help="Task type for prompt optimization (default: instruction)"
    )
    
    parser.add_argument(
        "--task-description",
        default="Generate helpful and accurate responses",
        help="Description of the task for prompt optimization"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum iterations for prompt optimization (default: 2)"
    )
    
    parser.add_argument(
        "--model-filter",
        help="Filter models by name (substring match) for optimization"
    )
    
    parser.add_argument(
        "--test-layout",
        action="store_true",
        help="Test the TUI layout and exit"
    )
    
    args = parser.parse_args()
    
    # Test layout mode
    if args.test_layout:
        setup_logging(args.log_level)
        context = SharedContext()
        tui = LMSAutoTUI(context)
        tui.log_output("[cyan]Testing layout...[/cyan]")
        tui.log_output("[green]Layout test complete![/green]")
        
        # Just show the layout for 3 seconds
        with Live(tui.layout, console=tui.console, screen=True) as live:
            for i in range(30):
                tui.update_layout()
                live.update(tui.layout)
                time.sleep(0.1)
        console.print("[green]Layout test completed successfully![/green]")
        return
    
    # Check if interactive mode should be used
    if (args.interactive or 
        not any([args.scan_only, args.profile_system, args.optimize_prompts, args.test_layout])):
        # Run TUI mode
        setup_logging(args.log_level)
        context = SharedContext()
        
        # Try TUI first, fallback to simple menu if it fails
        try:
            tui = LMSAutoTUI(context)
            tui.run()
        except Exception as e:
            console.print(f"[red]TUI failed: {e}[/red]")
            console.print("[yellow]Falling back to simple interactive mode...[/yellow]")
            interactive_mode(context)
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Display welcome banner
    console.print(Panel.fit(
        "[bold blue]LMSAuto[/bold blue] - LM Studio Autonomous Model Settings Optimizer",
        style="bold cyan"
    ))
    
    logger.info("Starting LMSAuto...")
    
    try:
        # Initialize shared context
        context = SharedContext()
        
        # Profile system hardware if requested
        if args.profile_system:
            with console.status("[bold green]Profiling system hardware..."):
                profiler = SystemProfiler(context)
                profiler.profile_system()
            
            # Display hardware specs with Rich table
            specs = context.get_hardware_specs()
            if specs:
                table = Table(title="System Hardware Specifications", box=box.ROUNDED)
                table.add_column("Component", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")
                
                for key, value in specs.items():
                    component = key.replace('_', ' ').title()
                    table.add_row(component, str(value))
                
                console.print(table)
                console.print()
        
        # Initialize and run model scanner
        with console.status("[bold green]Scanning for models..."):
            scanner = ModelScanner(context)
            lm_studio_scanner = LMStudioScanner()
            scanner.register_scanner(lm_studio_scanner)
            scanner.scan()
        
        # Display discovered models with Rich table
        models = context.get_all_models()
        if models:
            table = Table(title=f"Discovered Models ({len(models)})", box=box.ROUNDED)
            table.add_column("Model Name", style="cyan", no_wrap=True)
            table.add_column("Platform", style="green")
            table.add_column("Source", style="yellow")
            
            for model_key, model_info in models.items():
                source = model_info.metadata.get('source', 'unknown') if hasattr(model_info, 'metadata') and model_info.metadata else 'unknown'
                table.add_row(model_info.name, model_info.platform, source)
            
            console.print(table)
            console.print()
        else:
            console.print(Panel(
                "[bold red]No models found[/bold red]\n"
                "Make sure LM Studio is running with models loaded.",
                title="Warning",
                style="yellow"
            ))
            logger.warning("No models discovered. Check LM Studio installation and model availability.")
        
        if args.scan_only:
            logger.info("Scan-only mode complete.")
            return
        
        # Prompt optimization
        if args.optimize_prompts and models:
            console.print("\n[bold blue]Starting Prompt Optimization[/bold blue]")
            
            # Filter models if specified
            models_to_optimize = models
            if args.model_filter:
                models_to_optimize = {
                    key: model for key, model in models.items()
                    if args.model_filter.lower() in model.name.lower()
                }
                if not models_to_optimize:
                    console.print(f"[bold red]No models found matching filter: {args.model_filter}[/bold red]")
                    return
            
            # Limit to first 3 models to avoid long processing times
            if len(models_to_optimize) > 3:
                console.print(f"[yellow]Optimizing prompts for first 3 models (out of {len(models_to_optimize)} total)...[/yellow]")
                models_to_optimize = dict(list(models_to_optimize.items())[:3])
            
            # Initialize optimizer
            optimizer = PromptWizardOptimizer(context)
            
            # Optimize prompts for each model with progress bar
            results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[green]Optimizing prompts...", total=len(models_to_optimize))
                
                for model_key, model_info in models_to_optimize.items():
                    progress.update(task, description=f"[green]Optimizing: {model_info.name}")
                    
                    try:
                        result = optimizer.optimize(
                            model_key=model_key,
                            task_description=args.task_description,
                            task_type=args.task_type,
                            max_iterations=args.max_iterations
                        )
                        results.append(result)
                        
                        # Brief success message
                        console.print(f"[green]✅[/green] {model_info.name}: Improvement {result.improvement:.3f}")
                        
                    except Exception as e:
                        logger.error(f"Error optimizing prompt for {model_info.name}: {e}")
                        console.print(f"[red]❌[/red] {model_info.name}: Optimization failed")
                    
                    progress.advance(task)
            
            # Display summary with Rich table
            if results:
                # Summary table
                summary_table = Table(title="Prompt Optimization Results", box=box.ROUNDED)
                summary_table.add_column("Model", style="cyan")
                summary_table.add_column("Improvement", style="green")
                summary_table.add_column("Variants Tested", style="yellow")
                summary_table.add_column("Effectiveness", style="magenta")
                
                for result in results:
                    summary_table.add_row(
                        result.model_key.replace('lm_studio_', ''),
                        f"{result.improvement:.3f}",
                        str(result.variants_tested),
                        f"{result.metrics.get('overall_effectiveness', 0):.3f}"
                    )
                
                console.print(summary_table)
                
                # Show best performing optimization
                best_result = max(results, key=lambda r: r.metrics.get('overall_effectiveness', 0))
                
                console.print(Panel(
                    f"[bold green]Task:[/bold green] {args.task_description}\n"
                    f"[bold green]Task Type:[/bold green] {args.task_type}\n"
                    f"[bold green]Models Optimized:[/bold green] {len(results)}\n"
                    f"[bold green]Best Model:[/bold green] {best_result.model_key.replace('lm_studio_', '')}\n"
                    f"[bold green]Best Score:[/bold green] {best_result.metrics.get('overall_effectiveness', 0):.3f}",
                    title="Optimization Summary",
                    style="green"
                ))
                
                # Show the optimized prompt
                console.print(Panel(
                    best_result.optimized_prompt,
                    title="Best Optimized Prompt",
                    style="cyan"
                ))
        
        elif args.optimize_prompts and not models:
            console.print(Panel(
                "[bold red]No models found for prompt optimization[/bold red]\n"
                "Make sure LM Studio is running with models loaded.",
                title="Error",
                style="red"
            ))
        
        # TODO: Implement HuggingFace integration and config generation
        if not args.optimize_prompts:
            console.print(Panel(
                "[bold yellow]Configuration generation will be added in future updates[/bold yellow]\n"
                "Currently available:\n"
                "• Model discovery via LM Studio API\n"
                "• System hardware profiling\n"
                "• Advanced prompt optimization",
                title="Coming Soon",
                style="yellow"
            ))
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        console.print(Panel(
            f"[bold red]Error: {e}[/bold red]",
            title="Application Error",
            style="red"
        ))
        sys.exit(1)
    
    console.print("\n[bold green]✅ LMSAuto completed successfully![/bold green]")

if __name__ == "__main__":
    main()