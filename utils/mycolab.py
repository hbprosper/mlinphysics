# --------------------------------------------------------------------
# Download from https://github.com/hbprosper
# Created: Oct 03 2025 Introduction to ML course at AIMS South Africa
# Harrison B. Prosper
# --------------------------------------------------------------------
import os
import subprocess

CHECK = "\u2705"
FAIL  = "\u274C"
WARN  = "\u26A0"
# --------------------------------------------------------------------
def git_clone(folder='AIMS/tutorials/code',
              repo= f'https://github.com/hbprosper',
              name='mlinphysics'):
    
    url      = f'{repo}/{name}'
    raw_url  = url.replace('github.com', 'raw.githubusercontent.com')
    raw_url += '/refs/heads/main/'

    # 0. Make sure Google Drive is mounted
    if not os.path.ismount("/content/gdrive"):
        print(f"{FAIL} Google Drive is NOT mounted")
        print(' mount using google.colab.drive.mount("/content/gdrive")')
        return -1

    # 1. Move to the appropriate Google Drive folder
    os.chdir(f'/content/gdrive/MyDrive/{folder}')

    # 2. Delete mlinphysics folder
    result = subprocess.run(["rm", "-rf", name],
        capture_output=True)
    if result.returncode != 0:
        print(f"{FAIL} Deletion of {name} failed:",
              result.stderr.strip())
        return result.returncode

    # 3. Clone repo <name> with minimal history
    result = subprocess.run(
        ["git", "clone",
         "--depth", "1",
         "--filter=blob:none",
         "--sparse", url],
        capture_output=True,
        text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"{FAIL} Clone failed:", result.stderr.strip())
        return reseult.returncode
    
    # 4. Checkout folder utils
    os.chdir(name)
    result = subprocess.run(
        ["git", "sparse-checkout",
          "set", "utils"],
        capture_output=True,
        text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"{FAIL} Sparse checkout failed:", result.stderr.strip())
        return reseult.returncode
        
    for file in ['__init__.py', 'nn.py']:
        url = raw_url + file
        result = subprocess.run(
            ["wget", "-q", url, "-O", file],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{FAIL} wget of {file} failed:", result.stderr.strip())
            return reseult.returncode
            
    # 5. Remember to go back to code folder
    os.chdir("..")
 
    print(f"{CHECK} Download of '{name}' successful\n")