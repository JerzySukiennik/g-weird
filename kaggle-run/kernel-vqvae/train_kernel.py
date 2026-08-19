"""Bootstrap for G-Weird VQ-VAE training."""
import subprocess, sys
subprocess.run(["git","clone","--depth","1",
                "https://github.com/JerzySukiennik/g-weird.git","/tmp/gw"], check=True)
subprocess.run([sys.executable, "/tmp/gw/kaggle/02-train-vqvae.py"], check=True)
