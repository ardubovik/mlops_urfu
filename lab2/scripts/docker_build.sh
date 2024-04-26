#!/bin/bash

# create new image
docker build -t jdk-image ../.

# create and run container
docker run --restart always --name jenkins -p 8080:8080 -p 50000:50000 jdk-image:latest
