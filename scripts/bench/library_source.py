"""Resolve sources from the exact VeriTile checkout in the Lake lockfile."""
from functools import cache
import json
from pathlib import Path
import subprocess


@cache
def library_root(root: Path) -> Path:
    manifest = json.loads((root / 'lake-manifest.json').read_text())
    package, = (p for p in manifest['packages'] if p['name'] == 'veritile')
    path = root / manifest['packagesDir'] / package['name']
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != package['rev']:
        raise ValueError(f"Expected VeriTile revision {package['rev']}, found {head}.")
    dirty = subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--'], check=False)
    if dirty.returncode:
        raise ValueError('The locked VeriTile source checkout must be clean.')
    return path


def project_source(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Invalid benchmark source path: ' + name)
    if path.is_file():
        return path
    return dependency_source(root, name)


def dependency_source(root: Path, name: str) -> Path:
    dependency = library_root(root)
    path = (dependency / name).resolve()
    if not path.is_relative_to(dependency.resolve()):
        raise ValueError('Invalid VeriTile source path: ' + name)
    return path
