"""
Weavel, natural language analysis Dashboard for LLM Agent
"""

from setuptools import setup, find_namespace_packages

# Read README.md for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="weavel",
    version="1.6.0",
    packages=find_namespace_packages(),
    entry_points={},
    description="Weavel, natural language analysis Dashboard for LLM Agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="weavel",
    url="https://github.com/weavel-ai/weavel-python",
    install_requires=[
        "httpx[http2]",
        "pydantic>=2.4.2",
        "typer[all]",
        "pendulum",
        "requests",
        "cryptography",
        "pyyaml",
        "InquirerPy",
        "python-dotenv",
        "websockets",
        "termcolor",
        "watchdog",
        "readerwriterlock",
        "pendulum",
        "httpx[http2]",
        "nest_asyncio",
    ],
    python_requires=">=3.8.10",
    keywords=[
        "weavel",
        "agent",
        "llm",
        "llm evaluation",
        "llm monitoring",
        "prompt evaluation",
        "dataset curation",
        "funnel analytics",
    ],
)
