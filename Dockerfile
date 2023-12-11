# Use an official Python runtime as a parent image
FROM python:3.9-slim-bookworm

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Miniconda
ENV MINICONDA_VERSION latest
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    echo "export PATH=$PATH" >> ~/.bashrc

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Use the environment.yml to create the Conda environment.
# Replace 'your-environment.yml' with your actual environment file
COPY environment.yml /usr/src/app/environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rootsim", "/bin/bash", "-c"]

# Set the entry point to activate the conda environment
ENTRYPOINT ["conda", "run", "-n", "rootSim", "/bin/bash"]

# Activate the Conda environment and then wait for the user command
CMD ["source activate rootsim && tail -f /dev/null"]



