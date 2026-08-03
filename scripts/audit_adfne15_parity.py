#!/usr/bin/env python3
"""Audit ADFNE15 owner layout and optionally run its acceptance suite."""
from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys


def main():
    parser=ArgumentParser(); parser.add_argument('--mrst',type=Path,default=Path('../mrst-2024a-Geomech/modules/ADFNE15')); parser.add_argument('--run-tests',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; python_root=root/'pymrst/modules/adfne15'
    matlab={path.stem.lower() for path in args.mrst.glob('*.m')}
    owners={path.stem.lower().rstrip('_') for path in python_root.glob('*.py') if path.name!='__init__.py'}
    missing=sorted(matlab-owners); extras=sorted(owners-matlab)
    print(f'ADFNE15 MRST-layout owners: {len(matlab & owners)}/{len(matlab)}')
    print('Missing:', ', '.join(missing) if missing else 'none')
    print('Python extensions:', ', '.join(extras) if extras else 'none')
    if missing:return 1
    if args.run_tests:
        command=[sys.executable,'-m','pytest','-q','tests/adfne15']
        return subprocess.run(command,cwd=root).returncode
    return 0


if __name__=='__main__': raise SystemExit(main())
