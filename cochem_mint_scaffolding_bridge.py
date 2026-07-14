# Append this to CoChem-MInt.py

import os
import json
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print_status("CRITICAL: 'watchdog' library missing from main environment. Cannot build data bridge.", "fail")
    sys.exit(1)

# ---------------------------------------------------------
# STAGE 1.0: PROJECT SCAFFOLDING
# ---------------------------------------------------------
class ProjectScaffold:
    def __init__(self, project_name: str):
        self.project_name = f"CoChem-{project_name}"
        self.base_dir = os.path.abspath(self.project_name)
        self.input_dir = os.path.join(self.base_dir, "Input_Files")
        self.processed_dir = os.path.join(self.base_dir, "Processed")
        self.logs_dir = os.path.join(self.base_dir, "Logs")
        self.registry_path = os.path.join(self.processed_dir, "registry.json")

    def build_workspace(self) -> bool:
        """Constructs the rigid directory tree and initializes the local registry."""
        try:
            for directory in [self.input_dir, self.processed_dir, self.logs_dir]:
                os.makedirs(directory, exist_ok=True)
            
            if not os.path.exists(self.registry_path):
                initial_registry = {
                    "project_name": self.project_name,
                    "created_at": datetime.now().isoformat(),
                    "ingested_files": {},
                    "canonical_templates": {}
                }
                with open(self.registry_path, "w") as f:
                    json.dump(initial_registry, f, indent=4)
            
            print_status(f"Workspace configured at: {self.base_dir}", "success")
            return True
        except PermissionError:
            print_status(f"Permission denied creating workspace at {self.base_dir}", "fail")
            return False

# ---------------------------------------------------------
# STAGE 1.2: INTERACTIVE BRIDGE (BUILDAMOL)
# ---------------------------------------------------------
class MoleculeBuildHandler(FileSystemEventHandler):
    """Listens for new .xyz files saved by BuildAMol."""
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.xyz'):
            print_status(f"New geometry detected from builder: {os.path.basename(event.src_path)}", "info")
            self.callback(event.src_path)

class BuildAMolBridge:
    def __init__(self, silo_path: str, input_dir: str):
        self.silo_path = silo_path
        self.input_dir = input_dir
        self.python_bin = os.path.join(self.silo_path, "bin", "python")
        self.observer = None

    def _trigger_ingestion(self, file_path: str):
        """Callback fired when a new molecule is saved."""
        # This will link directly to Stage 2.0 (Intake & Hashing)
        print_status(f"Queuing {file_path} for canonical ingestion...", "success")

    def launch_and_watch(self):
        """Launches BuildAMol in a subprocess and watches the input directory."""
        if not os.path.exists(self.python_bin):
            print_status("BuildAMol Python binary missing. Silo may be corrupted.", "fail")
            return

        print_status("Initializing Data Bridge Watchdog...", "info")
        event_handler = MoleculeBuildHandler(self._trigger_ingestion)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_dir, recursive=False)
        self.observer.start()

        print_status("Launching BuildAMol interactive module...", "info")
        # Launching as a subprocess to keep the CoChem-MInt loop active
        env = os.environ.copy()
        # Ensure the subprocess saves directly to our project input folder
        env["WORKDIR"] = self.input_dir 
        
        try:
            # Command assumes BuildAMol can be invoked as a module or script
            subprocess.run(
                [self.python_bin, "-m", "buildamol"], 
                env=env, 
                cwd=self.input_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print_status(f"BuildAMol terminated unexpectedly: {e}", "warning")
        finally:
            print_status("Closing Data Bridge and halting Watchdog.", "info")
            self.observer.stop()
            self.observer.join()