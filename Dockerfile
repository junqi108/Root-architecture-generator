# Use an official Miniconda base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the environment.yml file to the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml
