#!/bin/bash
cmake -B build -S .
cmake --build build -t SuperPoint
./bin/SuperPoint 100