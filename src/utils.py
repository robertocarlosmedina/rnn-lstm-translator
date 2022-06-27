
import os
import zipfile
import shutil
from termcolor import colored


def download_crioleSet() -> None:
    try:
        os.system("wget https://github.com/robertocarlosmedina/crioleSet/archive/main.zip")
        zip_object = zipfile.ZipFile(f"main.zip", "r")
        zip_object.extractall(".data")
        os.rename(".data/crioleSet-main", ".data/crioleSet")
        os.remove(".data/crioleSet/main.py")
        os.remove(".data/crioleSet/README.md")
        os.remove(".data/crioleSet/RULES USED.txt")
        shutil.rmtree(".data/crioleSet/src")
        os.remove("main.zip")

        print(
            colored("==> The crioleSet dataset has been added to the project", attrs=["bold"]))
    except:
        print(
            colored("==> Error downloading the crioleSet dataset", "red", attrs=["bold"]))


def check_dataset() -> None:
    if not os.path.isdir(".data"):
        download_crioleSet()
    else: 
        print(colored("==> The crioleSet is in the project", attrs=["bold"]))

def progress_bar(value: int, max_width: int, display: str, unit: str, bar_size=20):
    bar_state = int((bar_size*value)/max_width)
    print(
        colored(f" [{'='*bar_state}>{' '*(bar_size-bar_state)}] {value}/{max_width} {unit}, {display}", attrs=["bold"]), 
        end='\r'
    )
