#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
QSAR Pipeline Intelligent Interface
===================================
This script serves as the central command center for the QSAR modeling pipeline.
It manages dependencies, checks file status, and guides the user through the
three-stage process:
1. Dataset Optimization
2. Model Training (XGBoost)
3. Prediction (New Compounds)

Author: Antigravity (Google DeepMind)
"""

import os
import sys
import glob
import subprocess
import time
import json
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
    'new_compounds_data': os.path.join(BASE_DIR, 'data', 'raw', 'new_compounds.xlsx'),
    'model_dir': os.path.join(BASE_DIR, 'results', 'model_metadata'),
    'predictions_dir': os.path.join(BASE_DIR, 'results', 'predictions'),
    
    # Scripts
    'script_opt': os.path.join(BASE_DIR, 'src', 'qsar_pipeline', 'dataset_optimizer.py'),
    'script_train': os.path.join(BASE_DIR, 'src', 'qsar_pipeline', 'xgboost_optimizer.py'),
    'script_pred': os.path.join(BASE_DIR, 'src', 'qsar_pipeline', 'molecular_predictor.py'),
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
    
    # helper for relative paths
    def rel(path): 
        return os.path.relpath(path, BASE_DIR)

    # 1. Dataset Optimization Status
    has_raw, raw_time_str, raw_ts = get_file_info(PATHS['raw_data'])
    has_opt, opt_time_str, opt_ts = get_file_info(PATHS['optimized_data'])
    
    state['step_1'] = {
        'name': "1. Dataset Optimization",
        'desc': "Cleans raw data, removes outliers, and selects best features.",
        'inputs': [f"📥 {rel(PATHS['raw_data'])}"],
        'outputs': [f"📤 {rel(PATHS['optimized_data'])}"],
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_raw:
        state['step_1']['status'] = "MISSING INPUT"
        state['step_1']['color'] = "red"
        state['step_1']['msg'] = "Raw file not found."
    elif not has_opt:
        state['step_1']['status'] = "READY TO START"
        state['step_1']['color'] = "yellow"
        state['step_1']['msg'] = "Ready to optimize."
    elif raw_ts > opt_ts:
        state['step_1']['status'] = "OUTDATED"
        state['step_1']['color'] = "orange1"
        state['step_1']['msg'] = "Raw data newer than optimized."
    else:
        state['step_1']['status'] = "COMPLETED"
        state['step_1']['color'] = "green"
        state['step_1']['msg'] = f"Last run: {opt_time_str}"

    # 2. Model Training Status
    has_model, model_name, model_ts = get_latest_model_info()
    
    state['step_2'] = {
        'name': "2. Model Training",
        'desc': "Trains XGBoost model with hyperparameter tuning.",
        'inputs': [f"📥 {rel(PATHS['optimized_data'])}"],
        'outputs': [f"📤 results/model_metadata/*.json"],
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_opt:
        state['step_2']['status'] = "BLOCKED"
        state['step_2']['color'] = "dim white"
        state['step_2']['msg'] = "Waiting for Step 1."
    elif not has_model:
        state['step_2']['status'] = "READY TO START"
        state['step_2']['color'] = "yellow"
        state['step_2']['msg'] = "Ready to train."
    elif opt_ts > model_ts:
        state['step_2']['status'] = "OUTDATED"
        state['step_2']['color'] = "orange1"
        state['step_2']['msg'] = "Data changed, re-train suggested."
    else:
        state['step_2']['status'] = "COMPLETED"
        state['step_2']['color'] = "green"
        state['step_2']['msg'] = f"Model: {model_name}"

    # 3. Prediction Status
    has_db, db_time_str, db_ts = get_file_info(PATHS['new_compounds_data'])
    
    state['step_3'] = {
        'name': "3. Prediction",
        'desc': "Predicts pIC50 for new compounds.",
        'inputs': [f"📥 {rel(PATHS['new_compounds_data'])}", "📥 Trained Model"],
        'outputs': [f"📤 results/predictions/"],
        'status': "UNKNOWN",
        'color': "white",
        'msg': ""
    }
    
    if not has_db:
        state['step_3']['status'] = "MISSING INPUT"
        state['step_3']['color'] = "red"
        state['step_3']['msg'] = "Input file not found."
    elif not has_model:
        state['step_3']['status'] = "BLOCKED"
        state['step_3']['color'] = "dim white"
        state['step_3']['msg'] = "Waiting for Step 2."
    else:
        state['step_3']['status'] = "READY"
        state['step_3']['color'] = "cyan"
        state['step_3']['msg'] = "Ready to predict."
        
    return state

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    title = Text("QSAR INTELLIGENT PIPELINE", justify="center", style="bold cyan")
    subtitle = Text("Dataset Optimization • XGBoost Modeling • New Compound Prediction", justify="center", style="dim white")
    console.print(Panel(title, subtitle=subtitle, border_style="cyan"))

def print_status_table(state):
    table = Table(title="Pipeline State", expand=True, border_style="dim", padding=(0,1))
    
    table.add_column("Step", style="bold cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    table.add_column("Input / Output", style="white")
    table.add_column("Status", justify="center")
    
    for step_key in ['step_1', 'step_2', 'step_3']:
        s = state[step_key]
        
        # Format IO
        io_text = "\n".join(s['inputs'] + s['outputs'])
        
        # Format Status
        status_text = f"[{s['color']}]{s['status']}[/]\n[dim]{s['msg']}[/]"
        
        table.add_row(
            s['name'],
            s['desc'],
            io_text,
            status_text
        )
    
    console.print(table)

# ==============================================================================
# ACTIONS
# ==============================================================================
def run_script(script_path_or_cmd, description):
    """Executes a python script with a nice spinner."""
    # Determine actual script path for verification
    if isinstance(script_path_or_cmd, list):
        script_path = script_path_or_cmd[0]
        cmd = [sys.executable] + script_path_or_cmd
    else:
        script_path = script_path_or_cmd
        cmd = [sys.executable, script_path]

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
    
    console.rule(f"[bold cyan]Executing: {description}[/]")
    
    
    # Handle list of arguments (script path + args) - ALREADY DONE ABOVE
    
    try:
        # Simple run allowing output to flow to terminal so user sees progress bars of the scripts
        result = subprocess.run(cmd, check=True)
        
        duration = time.time() - start_time
        console.print(f"\n[bold green]✓ {description} Completed successfully in {duration:.1f}s[/]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]✕ Error executing {description}. Exit code: {e.returncode}[/]")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✕ Unexpected error: {e}[/]")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✕ Unexpected error: {e}[/]")
        return False

def get_step_2_grid_config():
    """Prompts user for Step 2 Grid Configuration and returns flags."""
    console.print("\n[bold cyan]Step 2 Configuration: Hyperparameter Grid Search[/]")
    console.print("  [1] [green]Default (Exhaustive)[/]: Best for final models. [yellow](Time Consuming)[/]")
    console.print("  [2] [yellow]Fast (Verification)[/]: Best for testing/debugging. [dim](Quick)[/]")
    console.print("  [3] [blue]Custom (File)[/]: Load from 'grid_config.json'.")
    
    choice = Prompt.ask("\nSelect configuration", choices=["1", "2", "3"], default="1")
    
    if choice == '1':
        return [] # No flags = Default
        
    elif choice == '2':
        # Create temporary fast config
        fast_grid = {
            'n_estimators': [10], 
            'max_depth': [3]
        }
        fast_config_path = os.path.join(BASE_DIR, 'fast_grid_temp.json')
        try:
            with open(fast_config_path, 'w') as f:
                json.dump(fast_grid, f)
            return ['--config', fast_config_path]
        except Exception as e:
            console.print(f"[red]Error creating fast config: {e}[/]")
            return []
            
    elif choice == '3':
        custom_path = os.path.join(BASE_DIR, 'src', 'qsar_pipeline', 'grid_config.json')
        if not os.path.exists(custom_path):
            console.print(f"[red]Error: Custom file not found at {custom_path}. Using Default.[/]")
            return []
        return ['--config', custom_path]
        
    return []

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
            
            # Get Config
            grid_args = get_step_2_grid_config()
            
            # Combine script path and args
            cmd_args = [PATHS['script_train']] + grid_args
            run_script(cmd_args, "XGBoost Training")
            Prompt.ask("\n[dim]Press Enter to continue...[/]")
            
        elif choice == '3':
            # Dependency Check
            if state['step_3']['status'] == "BLOCKED":
                if not Confirm.ask("[yellow]Warning: Model is missing. Proceed anyway?[/yellow]"):
                    continue
            run_script(PATHS['script_pred'], "New Compounds Prediction")
            Prompt.ask("\n[dim]Press Enter to continue...[/]")
            
        elif choice.lower() == 'a':
            if Confirm.ask("This will run all 3 steps in sequence. Continue?"):
                # Ask for Step 2 config upfront
                grid_args = get_step_2_grid_config()
                step_2_cmd = [PATHS['script_train']] + grid_args
                
                if run_script(PATHS['script_opt'], "Step 1: Optimization"):
                    if run_script(step_2_cmd, "Step 2: Training"):
                        run_script(PATHS['script_pred'], "Step 3: Prediction")
                Prompt.ask("\n[dim]Press Enter to continue...[/]")

if __name__ == "__main__":
    main()
