<div align="center">
    <a href="https://www.weavel.ai">
        <img src="https://i.imgur.com/uQ7ulX3.png" title="Logo" style="" />
    </a>
    <h1>Weavel Python SDK</h1>
    <h3>Prompt Optimization & Evaluation for LLM Applications</h3>
    <div>
        <a href="https://pypi.org/project/weavel" target="_blank">
            <img src="https://img.shields.io/pypi/v/weavel.svg" alt="PyPI Version"/>
        </a>
    </div>
</div>

## Installation

```bash
pip install weavel
```

## Documentation

You can find our full documentation [here](https://weavel.ai/docs/python-sdk).

## How to use

### Option 1: Using OpenAI wrapper

```python
from weavel import WeavelOpenAI as OpenAI

openai = OpenAI()

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    headers={
      "generation_name": "hello",
    }
)

```

### Option 2: Logging inputs/outputs of LLM calls

```python
from weavel import Weavel
from openai import OpenAI
from pydantic import BaseModel

openai = OpenAI()
# initialize Weavel
weavel = Weavel()

class Answer(BaseModel):
    reasoning: str
    answer: str

question = "What is x if x + 2 = 4?"
response = openai.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a math teacher."},
        {"role": "user", "content": question}
    ],
    response_format=Answer
).choices[0].message.parsed

# log the generation
weavel.generation(
    name="solve-math", # optional
    inputs={"question": question},
    outputs=response.model_dump()
)
```

### Option 3 (Advanced Usage): OTEL-compatible trace logging

```python
from weavel import Weavel

weavel = Weavel()

session = weavel.session(user_id = "UNIQUE_USER_ID")

session.message(
    role="user",
    content="Nice to meet you!"
)

session.track(
    name="Main Page Viewed"
)

trace = session.trace(
    name="retrieval_module"
)

trace.log(
    name="google_search"
)
```
