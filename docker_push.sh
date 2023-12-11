#!/usr/bin/env bash

# Load environment variables
. .env

###################################################################
# Main
###################################################################

# Login to GitHub Container Registry
echo $CR_PAT | sudo docker login ghcr.io -u USERNAME --password-stdin

# Find DOCKER_IMAGE_HASH of the rootsim_container
# This command filters the images to find the one named "rootsim" and retrieves its ID
IFS='\n' read -r -a array <<< "$(sudo docker images --filter=reference='rootsim:latest' --format '{{.ID}}')"
echo "${array[0]}"

# Tag the image for the GitHub Container Registry
# Replace CONTAINER_REG_USER and CONTAINER_REG_NAME with your GitHub username and desired repository name
docker tag "${array[0]}" "ghcr.io/${CONTAINER_REG_USER}/${CONTAINER_REG_NAME}:latest"

# Push the image to the GitHub Container Registry
docker push "ghcr.io/${CONTAINER_REG_USER}/${CONTAINER_REG_NAME}:latest"

