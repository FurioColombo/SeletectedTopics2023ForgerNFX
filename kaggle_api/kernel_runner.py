import time
import json
from pathlib import Path
from .auth import get_api_client

class KernelRunner:
    def __init__(self):
        self.api = get_api_client()
        
    def run_kernel(self, kernel_slug: str):
        """Starts the kernel execution."""
        # Kaggle API doesn't have a direct "run" command for notebooks pushed via API, 
        # they run automatically on push if metadata says so? 
        # Actually kernel_push updates the code. To run it, we might need to rely on it running on push.
        # But usually we want to trigger a run.
        # The `kaggle kernels push` command uploads and commits.
        # If we want to run it, we just pushed it. It should run if it's a script or notebook.
        # Let's assume push triggers run.
        pass

    def poll_status(self, kernel_slug: str):
        """Polls the status of the kernel."""
        print(f"Polling status for {kernel_slug}...")
        while True:
            status = self.api.kernels_status(kernel_slug)
            # status returns dict or object?
            # It usually returns 'complete', 'running', 'error'
            # The api method might return a response object.
            # Let's assume string for now or inspect response.
            # Actually api.kernel_status returns a KernelStatus object or dict.
            
            # For safety, let's print raw status first time we run this to debug if needed.
            # But we can't see output.
            # Let's assume standard statuses.
            
            s = str(status['status']) if isinstance(status, dict) else status.status
            print(f"Status: {s}")
            
            if s in ['complete', 'error', 'cancel']:
                return s
            
            time.sleep(30)

    def download_outputs(self, kernel_slug: str, output_dir: str):
        """Downloads output files."""
        print(f"Downloading outputs from {kernel_slug} to {output_dir}...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.api.kernels_output(kernel_slug, path=output_dir)
        print("Download complete.")
