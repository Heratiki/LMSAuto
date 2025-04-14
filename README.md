# LMSAuto - LM Studio Autonomous Model Settings Optimizer

LMSAuto is a Python tool designed to automatically discover models available in your local LM Studio instance, find potentially optimal configuration settings for them by searching Hugging Face, and generate configuration profiles. The goal is to simplify the process of optimizing model performance and usability within LM Studio.

## Features

- Discover models currently loaded or available in LM Studio (via API).
- Search Hugging Face Hub for configuration files (`generation_config.json`, `config.json`) associated with discovered models. Possible will need to be generated, scraped from the site, or inferred.
- Generate JSON profiles containing potential optimal settings.
- (Planned) Provide a Rich Terminal UI to manage and apply these profiles.
- (Planned) Automate applying selected settings to models in LM Studio.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/LMSAuto.git # Replace with actual repo URL
cd LMSAuto

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --help

# Example: Run discovery and profile generation
python -m src.main
```

*(Further usage instructions and details about the Rich UI will be added here.)*

## Configuration

*(Details about configuration files or environment variables will be added here.)*

## Contributing

*(Contribution guidelines will be added here.)*

## License

*(License information will be added here - currently using WORKSPACE-LICENSE)*