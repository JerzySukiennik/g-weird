"""Bootstrap for encoding the G-Weird corpus into tokens."""
import subprocess, sys
subprocess.run(["git","clone","--depth","1",
                "https://github.com/JerzySukiennik/g-weird.git","/tmp/gw"], check=True)
subprocess.run([sys.executable, "/tmp/gw/kaggle/03-encode.py"], check=True)
