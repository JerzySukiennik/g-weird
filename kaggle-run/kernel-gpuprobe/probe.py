"""Ten-second GPU probe: does this account still have weekly quota, and what card."""
import subprocess, torch
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout, flush=True)
print("cuda:", torch.cuda.is_available(), "| kart:", torch.cuda.device_count(), flush=True)
