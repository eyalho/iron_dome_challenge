import glob
import os
import pathlib
import shutil


def copy_py_files(src, dest):
    for file_path in glob.glob(os.path.join(src, '**', '*.py'), recursive=True):
        new_path = os.path.join(dest, os.path.basename(file_path))
        shutil.copy(file_path, new_path)


def save_program_files(debug, dst_folder):
    src = pathlib.Path(__file__).parent.parent.absolute()
    dst_folder = os.path.join(src, dst_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    debug("copy {src} to {dst}")
    copy_py_files(src, dst_folder)
