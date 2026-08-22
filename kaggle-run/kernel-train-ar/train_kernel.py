"""Bootstrap for G-Weird transformer training."""
import subprocess, sys
subprocess.run(["git","clone","--depth","1",
                "https://github.com/JerzySukiennik/g-weird.git","/tmp/gw"], check=True)
subprocess.run([sys.executable, "/tmp/gw/kaggle/04-train-ar.py"], check=True)
