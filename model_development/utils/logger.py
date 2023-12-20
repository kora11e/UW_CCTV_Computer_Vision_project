import os
from pathlib import Path
import datetime

class Logger:
    def __init__(self, class_name:str, instance_id):
        self.log_directory_parent = f"./{class_name} Instance Logs"
        
        self.log_directory = f"{self.log_directory_parent}/{class_name}_{instance_id}"
        
    def log(self, content):
        logs_directory = Path(self.log_directory)
        current_time = datetime.datetime.now().strftime("%-H%M")
        
        log_file_path = logs_directory / f"logs_{datetime.date.today().strftime('%-d-%m-%Y')}.txt"
        
        logs_directory.mkdir(parents=True, exist_ok=True)

        with open(log_file_path, 'a') as log_file:
            log_file.write(f"[{current_time}] {content}\n")