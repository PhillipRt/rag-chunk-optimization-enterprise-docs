#!/bin/bash
set -e

echo -e "\e[32mSetting up Python environment for RAG evaluation project...\e[0m"

if ! command -v uv &> /dev/null; then
    echo -e "\e[31muv is not installed. Install with:\e[0m"
    echo -e "\e[33mpip install uv\e[0m"
    exit 1
fi

UV_VERSION=$(uv --version)
echo -e "\e[32mFound uv version: $UV_VERSION\e[0m"

# Define a consistent venv path outside Windows mount
PROJECT_NAME=$(basename "$(pwd)")
VENV_DIR="$HOME/.venvs/${PROJECT_NAME}-venv"

echo -e "\e[32mCreating virtual environment at ${VENV_DIR} (link-mode: copy)...\e[0m"
uv venv --link-mode copy "$VENV_DIR"

echo -e "\e[32mActivating virtual environment...\e[0m"
source "${VENV_DIR}/bin/activate"

echo -e "\e[32mInstalling dependencies from requirements.txt...\e[0m"
uv pip install -r requirements.txt

echo -e "\e[32mEnvironment setup complete!\e[0m"
echo -e "\e[36mActivate this environment anytime with:\e[0m"
echo -e "\e[33msource ${VENV_DIR}/bin/activate\e[0m"
