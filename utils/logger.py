import os
import sys
import time
from pathlib import Path


def touch_file_if_missing(log_dir, log_name):
    # create dir & file
    dir_path = Path(log_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / log_name
    file_path.touch(exist_ok=True)

    return file_path

class Logger():
    """Write console output to external text file.
    
    Args:
        fpath (str): directory to save logging file
    """

    def __init__(self, file_path):
        self.console = sys.stdout
        self.file = open(file_path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        self.file.close()

def setup_logger(output_dir, output_name):
    # create dir
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # rename
    # file_stem, file_suffix = output_name.split('.')
    # output_new_name = file_stem + time.strftime("-%Y-%m-%d-%H-%M-%S") + '.' + file_suffix
    # file_path = dir_path / output_new_name

    # create file
    file_path = dir_path / output_name
    file_path.touch(exist_ok=True)

    sys.stdout = Logger(file_path)