import os
import sys

def delete_files(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('tmp') or filename.endswith('log') or filename.endswith('obj') or filename.endswith('txt'):
                file_path = os.path.join(foldername, filename)
                try:
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
                        print(f'Deleted file: {file_path}')
                except OSError as e:
                    print(f'Error: {file_path} : {e.strerror}')

directory = sys.argv[1]
delete_files(directory)

