#!/bin/sh
set -eu
python3 -c 'from pymrst.research_studio.case import read_reservoir_case; c=read_reservoir_case('"'"'cases/spe1.case.json'"'"'); c.construct(); print(c.content_hash)'
