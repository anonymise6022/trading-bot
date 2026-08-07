import sys
import subprocess

# List the exact libraries you need here
LIBRARIES = [
    "",
    "",
    ""
]

def install_or_update_packages():
    print("Checking and updating project libraries...\n")
    
    for library in LIBRARIES:
        print(f"--- Processing {library} ---")
        try:
            # --upgrade installs if missing, or updates to the newest stable version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", library])
        except subprocess.CalledProcessError as e:
            print(f"Error processing {library}: {e}")
            
    print("\nAll libraries are up to date!")

if __name__ == "__main__":
    install_or_update_packages()

