#!/Users/luisseijas/miniforge3/envs/IC-50/bin/python
# -*- coding: utf-8 -*-

"""
QSAR Pipeline Intelligent Interface
===================================
This script serves as the central command center for the QSAR modeling pipeline.
It manages dependencies, checks file status, and guides the user through the
three-stage process:
1. Dataset Optimization
2. Model Training (XGBoost)
3. Prediction (DrugBank)

Author: Antigravity (Google DeepMind)
"""

import os
import sys
import glob
import subprocess
import time
from datetime import datetime

# Try to import rich for beautiful UI, otherwise prompt to install
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.status import Status
    from rich import print as rprint
except ImportError:
    print("This interface requires the 'rich' library for its UI.")
    print("Please install it running: pip install rich")
    sys.exit(1)

# Initialize Console
console = Console()

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output definitions for State Tracking
PATHS = {
    'raw_data': os.path.join(BASE_DIR, 'data', 'raw', 'all_descriptor_results_1751.xlsx'),
    'optimized_data': os.path.join(BASE_DIR, 'data', 'processed', 'dataset_molecular_optimizado.xlsx'),
    'drugbank_data': os.path.join(BASE_DIR, 'data', 'raw', 'drugbank_compounds_cleaned.xlsx'),
    'model_dir': os.path.join(BASE_DIR, 'results', 'model_metadata'),
    'predictions_dir': os.path.join(BASE_DIR, 'results', 'predictions'),
    
    # Scripts
    'script_opt': os.path.join(BASE_DIR, 'dataset_optimizer.py'),
    'script_train': os.path.join(BASE_DIR, 'xgboost_optimizer.py'),
    'script_pred': os.path.join(BASE_DIR, 'drugbank_predictor.py'),
}

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
def get_file_info(path):
    """Returns exists (bool) and modification time (str/obj)"""
    if not os.path.exists(path):
        return False, None, 0
    mtime = os.path.getmtime(path)
    return True, datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'), mtime

def get_latest_model_info():
    """Finds the most recent model in the metadata directory."""
    if not os.path.exists(PATHS['model_dir']):
        return False, None, 0
    
    # Look for model JSONs
    model_files = glob.glob(os.path.join(PATHS['model_dir'], "modelo_xgboost_*.json"))
    if not model_files:
        return False, None, 0
        
    latest_model = max(model_files, key=os.path.getmtime)
    mtime = os.path.getmtime(latest_model)
    return True, os.path.basename(latest_model), mtime

def check_pipeline_state():
    """
    Analyzes the project state and returns a dictionary with status flags.
    STATUS_CODES: 
    0: Ready/Done
    1: Missing Input
    2: Outdated (Needs re-run)
    3: Not Started
    """
    state = {}
    
    # 1. Dataset Optimization Status
    has_raw, raw_time_str, raw_ts = get_file_info(PATHS['raw_data'])
    has_opt, opt_time_str, opt_ts = get_file_info(PATHS['optimized_data'])
    
    state['step_1'] = {
        'name': "Dataset Optimization",
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_raw:
        state['step_1']['status'] = "MISSING INPUT"
        state['step_1']['color'] = "red"
        state['step_1']['msg'] = "Raw file 'all_descriptor_results_1751.xlsx' not found."
    elif not has_opt:
        state['step_1']['status'] = "READY TO START"
        state['step_1']['color'] = "yellow"
        state['step_1']['msg'] = "Raw data available. Optimization needed."
    elif raw_ts > opt_ts:
        state['step_1']['status'] = "OUTDATED"
        state['step_1']['color'] = "orange1"
        state['step_1']['msg'] = "Raw data is newer than optimized data."
    else:
        state['step_1']['status'] = "COMPLETED"
        state['step_1']['color'] = "green"
        state['step_1']['msg'] = f"Optimized data ready ({opt_time_str})."

    # 2. Model Training Status
    has_model, model_name, model_ts = get_latest_model_info()
    
    state['step_2'] = {
        'name': "Model Training",
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_opt:
        state['step_2']['status'] = "BLOCKED"
        state['step_2']['color'] = "dim white"
        state['step_2']['msg'] = "Waiting for optimized dataset."
    elif not has_model:
        state['step_2']['status'] = "READY TO START"
        state['step_2']['color'] = "yellow"
        state['step_2']['msg'] = "Dataset ready. No model trained yet."
    elif opt_ts > model_ts:
        state['step_2']['status'] = "OUTDATED"
        state['step_2']['color'] = "orange1"
        state['step_2']['msg'] = "Dataset changed since last training."
    else:
        state['step_2']['status'] = "COMPLETED"
        state['step_2']['color'] = "green"
        state['step_2']['msg'] = f"Model ready ({model_name})."

    # 3. Prediction Status
    has_db, db_time_str, db_ts = get_file_info(PATHS['drugbank_data'])
    
    state['step_3'] = {
        'name': "Prediction (DrugBank)",
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_db:
        state['step_3']['status'] = "MISSING INPUT"
        state['step_3']['color'] = "red"
        state['step_3']['msg'] = "DrugBank file not found."
    elif not has_model:
        state['step_3']['status'] = "BLOCKED"
        state['step_3']['color'] = "dim white"
        state['step_3']['msg'] = "Waiting for trained model."
    else:
        state['step_3']['status'] = "READY"
        state['step_3']['color'] = "cyan"
        state['step_3']['msg'] = "Ready to generate predictions."
        
    return state

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    title = Text("QSAR INTELLIGENT PIPELINE", justify="center", style="bold cyan")
    subtitle = Text("Dataset Optimization • XGBoost Modeling • DrugBank Prediction", justify="center", style="dim white")
    console.print(Panel(title, subtitle=subtitle, border_style="cyan"))

def print_status_table(state):
    table = Table(title="Pipeline State", expand=True, border_style="dim")
    
    table.add_column("Step", style="bold")
    table.add_column("Current Status", justify="center")
    table.add_column("Details")
    
    table.add_row(
        "1. Optimize Data", 
        f"[{state['step_1']['color']}]{state['step_1']['status']}[/]", 
        state['step_1']['msg']
    )
    table.add_row(
        "2. Train Model", 
        f"[{state['step_2']['color']}]{state['step_2']['status']}[/]", 
        state['step_2']['msg']
    )
    table.add_row(
        "3. Predict", 
        f"[{state['step_3']['color']}]{state['step_3']['status']}[/]", 
        state['step_3']['msg']
    )
    
    console.print(table)

# ==============================================================================
# ACTIONS
# ==============================================================================
def run_script(script_path, description):
    """Executes a python script with a nice spinner."""
    if not os.path.exists(script_path):
        console.print(f"[red]Error: Script {script_path} not found![/]")
        return False
        
    start_time = time.time()
    
    # We use subprocess to run it. 
    # capturing output to show success/fail, but maybe streaming is better if long?
    # For this UI, let's stream if possible or just show spinner.
    # Given the previous scripts print a lot, let's just run it and let it print to a subprocess 
    # but we wrap it visually.
    
    console.rule(f"[bold cyan]Executing: {description}[/]")
    
    try:
        # Simple run allowing output to flow to terminal so user sees progress bars of the scripts
        result = subprocess.run([sys.executable, script_path], check=True)
        
        duration = time.time() - start_time
        console.print(f"\n[bold green]✓ {description} Completed successfully in {duration:.1f}s[/]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]✕ Error executing {description}. Exit code: {e.returncode}[/]")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✕ Unexpected error: {e}[/]")
        return False

# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    while True:
        clear_screen()
        print_header()
        
        state = check_pipeline_state()
        print_status_table(state)
        
        console.print("\n[bold]Available Actions:[/]")
        console.print("  [1] [cyan]Run Step 1:[/cyan] Optimize Dataset")
        console.print("  [2] [cyan]Run Step 2:[/cyan] Train XGBoost Model")
        console.print("  [3] [cyan]Run Step 3:[/cyan] Predict New Compounds")
        console.print("  [A] [magenta]Run Full Pipeline (1 -> 2 -> 3)[/magenta]")
        console.print("  [Q] Quit")
        
        choice = Prompt.ask("\nSelect an action", choices=["1", "2", "3", "A", "a", "Q", "q"], default="A")
        
        if choice.lower() == 'q':
            console.print("[dim]Goodbye![/]")
            break
            
        elif choice == '1':
            run_script(PATHS['script_opt'], "Dataset Optimization")
            Prompt.ask("\n[dim]Press Enter to continue...[/]")
            
        elif choice == '2':
            # Dependency Check
            if state['step_2']['status'] == "BLOCKED":
                if not Confirm.ask("[yellow]Warning: Step 1 data is missing. Proceed anyway?[/yellow]"):
                    continue
            run_script(PATHS['script_train'], "XGBoost Training")
            Prompt.ask("\n[dim]Press Enter to continue...[/]")
            
        elif choice == '3':
            # Dependency Check
            if state['step_3']['status'] == "BLOCKED":
                if not Confirm.ask("[yellow]Warning: Model is missing. Proceed anyway?[/yellow]"):
                    continue
            run_script(PATHS['script_pred'], "DrugBank Prediction")
            Prompt.ask("\n[dim]Press Enter to continue...[/]")
            
        elif choice.lower() == 'a':
            if Confirm.ask("This will run all 3 steps in sequence. Continue?"):
                if run_script(PATHS['script_opt'], "Step 1: Optimization"):
                    if run_script(PATHS['script_train'], "Step 2: Training"):
                        run_script(PATHS['script_pred'], "Step 3: Prediction")
                Prompt.ask("\n[dim]Press Enter to continue...[/]")

if __name__ == "__main__":
    main()
