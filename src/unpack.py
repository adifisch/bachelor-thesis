from os import walk
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

dataDir = 'data/faq-data/'

def unpackFile(target, unpack_dir):
    if target.endswith(".tgz"):
        tar = tarfile.open(target, "r")
        tar.extractall(unpack_dir)
        tar.close()
    elif target.endswith("tar"):
        tar = tarfile.open(target, "r")
        tar.extractall(unpack_dir)
        tar.close()

files = []
for (dirpath, dirnames, filenames) in walk(dataDir):
    files.extend(filenames)
    with ThreadPoolExecutor() as executor:
        for file in files:
            executor.submit(unpackFile, dataDir + file, 'data/unpack-faq/')