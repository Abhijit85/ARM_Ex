pip install numpy
pip install llama-index-vector-stores-faiss
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-langchain
pip install llama-index
pip install langchain-community
pip install llama-index-embeddings-langchain
pip install -q -U google-generativeai
pip install llama-index-llms-gemini
pip install llama-index-retrievers-bm25
pip install llama-index-embeddings-gemini
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-voyageai
pip install llama-index-vector-stores-chroma
pip install llama-index-embeddings-google-genai
pip install llama-index-postprocessor-cohere-rerank
pip install llama-index-llms-anthropic
pip install llama-index-llms-ollama

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4


export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install faiss-gpu-cu12
pip install einops
pip uninstall bitsandbytes -y