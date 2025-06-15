#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE_MODEL file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
set -e

# Help function
show_help() {
    local is_error=${1:-true}  # Default to error mode if no argument provided
    
    echo "Usage: $0 --model <model_size> --dest <destination_directory>"
    echo
    echo "Required arguments:"
    echo "  --model <model_size>    Size of the model to download"
    echo "  --dest <directory>      Directory where the model will be downloaded"
    echo
    echo "Available model sizes:"
    echo "  0.5b  - 0.5B parameter model (FP16)"
    echo "  1.5b  - 1.5B parameter model (INT8)"
    echo "  7b    - 7B parameter model (INT4)"
    echo
    echo "Options:"
    echo "  --help    Show help message"
    
    # Exit with success (0) for help flag, error (1) for usage errors
    if [ "$is_error" = "false" ]; then
        exit 0
    else
        exit 1
    fi
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model_size="$2"; shift ;;
        --dest) dest_dir="$2"; shift ;;
        --help) show_help false ;;  # Explicit help request
        *) echo -e "Unknown parameter: $1\n"; show_help true ;;  # Error case
    esac
    shift
done

# Validate required parameters
if [ -z "$model_size" ]; then
    echo -e "Error: --model parameter is required\n"
    show_help true
fi

if [ -z "$dest_dir" ]; then
    echo -e "Error: --dest parameter is required\n"
    show_help true
fi

# Map model size to full model name
case "$model_size" in
    "0.5b") model="llava-fastvithd_0.5b_stage3_llm.fp16" ;;
    "1.5b") model="llava-fastvithd_1.5b_stage3_llm.int8" ;;
    "7b") model="llava-fastvithd_7b_stage3_llm.int4" ;;
    *)
        echo -e "Error: Invalid model size '$model_size'\n"
        show_help true
        ;;
esac

cleanup() { 
    rm -rf "$tmp_dir"
}

download_model() {
    # Download directory
    tmp_dir=$(mktemp -d)

    # Model paths
    base_url="https://ml-site.cdn-apple.com/datasets/fastvlm"

    # Create destination directory if it doesn't exist
    if [ ! -d "$dest_dir" ]; then
        echo "Creating destination directory: $dest_dir"
        mkdir -p "$dest_dir"
    elif [ "$(ls -A "$dest_dir")" ]; then
        echo -e "Destination directory '$dest_dir' exists and is not empty.\n"
        read -p "Do you want to clear it and continue? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo -e "\nStopping."
            exit 1
        fi
        echo -e "\nClearing existing contents in '$dest_dir'"
        rm -rf "${dest_dir:?}"/*
    fi

    # Create temp variables
    tmp_zip_file="${tmp_dir}/${model}.zip"
    tmp_extract_dir="${tmp_dir}/${model}"

    # Create temp extract directory
    mkdir -p "$tmp_extract_dir"

    # Download model
    echo -e "\nDownloading '${model}' model ...\n"
    curl --progress-bar -L -o "$tmp_zip_file" "$base_url/$model.zip"

    # Unzip model
    echo -e "\nUnzipping model..."
    unzip -q "$tmp_zip_file" -d "$tmp_extract_dir"

    # Copy model files to destination directory
    echo -e "\nCopying model files to destination directory..."
    cp -r "$tmp_extract_dir/$model"/* "$dest_dir"

    # Verify destination directory exists and is not empty
    if [ ! -d "$dest_dir" ] || [ -z "$(ls -A "$dest_dir")" ]; then
        echo -e "\nModel extraction failed. Destination directory '$dest_dir' is missing or empty."
        exit 1
    fi

    echo -e "\nModel downloaded and extracted to '$dest_dir'"
}

# Cleanup download directory on exit
trap cleanup EXIT INT TERM

# Download models
download_model 