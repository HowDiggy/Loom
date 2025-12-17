# Loom

**Loom** is a local Python client for high-performance Large Language Model (LLM) inference. It is designed to act as the interface between data pipelines (ETL) and the self-hosted inference engine running on the local Threadripper workstation ("Lenny").

It provides a strict, type-safe wrapper around the OpenAI API standard, enforcing **Design by Contract** principles and guaranteeing structured JSON outputs via Pydantic schema validation.

## Features
* **Zero-Cost Inference:** Uses local hardware (Threadripper Pro 3945W).
* **Structured Data Extraction:** Guarantees valid JSON output matching Pydantic models.
* **Type Safety:** Full Python typing and rigorous pre/post-condition checks.
* **Infrastructure-Aware:** specifically tuned for the `gpt-oss-120b` model.

---

## Installation

Loom is managed via Poetry. It is designed to be installed as a library into other projects (like `job_hunter`).

```bash
# Install as a dependency in another project
poetry add --editable /path/to/loom
Usage
1. Simple Generation

Python
from loom.client import LoomClient

client = LoomClient()
response = client.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain the benefits of AVX-512 instructions."
)
print(response.content)
2. Structured Extraction (ETL)

Loom shines at extracting structured data from unstructured text.

Python
from pydantic import BaseModel
from loom.client import LoomClient

class JobListing(BaseModel):
    title: str
    is_remote: bool
    salary_max: int

client = LoomClient()
result = client.generate_structured(
    system_prompt="Extract job details.",
    user_prompt="We are looking for a Remote Python Engineer. Pay up to $150k.",
    response_model=JobListing
)

# result is a guaranteed instance of JobListing
print(result.is_remote) # True
Infrastructure: "Lenny" Configuration
This library relies on a specific backend configuration running on the workstation Lenny (192.168.1.6).

Hardware Spec

Host: Lenovo P620

CPU: AMD Ryzen Threadripper PRO 3945W

RAM: 128GB DDR4

OS: Ubuntu Server (Headless)

Model Details

Model: gpt-oss-120b

Quantization: UD-Q8_K_XL (Unsloth Dynamic 8-bit)

Path: /data/models/UD-Q8_K_XL/gpt-oss-120b-UD-Q8_K_XL-00001-of-00002.gguf

Service Configuration (Systemd)

The inference engine runs as a systemd service called llama-server.

Location: /etc/systemd/system/llama-server.service

Configuration:

Ini, TOML
[Unit]
Description=Llama.cpp API Server (GPT-OSS-120B)
After=network.target

[Service]
User=paulo
Group=paulo
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m /data/models/UD-Q8_K_XL/gpt-oss-120b-UD-Q8_K_XL-00001-of-00002.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 32768 \
    -t 24 \
    -cb \
    -np 4 \
    --alias gpt-120b
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
Maintenance Commands

If the server needs a restart or logs need checking:

Bash
# Check status
sudo systemctl status llama-server

# View logs
journalctl -u llama-server -f

# Restart service
sudo systemctl restart llama-server
