#!/bin/bash
python imaging.py
if [ $? -eq 0 ]; then
  feh output.png earth.png &
fi
