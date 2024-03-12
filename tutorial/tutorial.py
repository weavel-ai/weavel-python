"""Tutorial code for Weavel Python SDK."""
from fastapi import FastAPI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from weavel import create_client # 1. Import the weavel package (pip install weavel)

load_dotenv() # 2. Set Environment variable WEAVEL_API_KEY

app = FastAPI()
openai_client = OpenAI()
weavel_client = create_client() # 3. Create a Weavel client

origins = ["*"]

@app.post("/chat")
async def chat(
    messages: List[Dict[str, Any]], 
):
    """Chat with GPT-3.5-Turbo"""
    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )
    
    return response.choices[0].message.content


@app.post("/chat_with_logging")
async def chat_with_logging(
    messages: List[Dict[str, Any]], 
    user_uuid: Optional[str] = None,
    trace_uuid: Optional[str] = None
):
    """Chat with GPT-3.5-Turbo"""
    if not trace_uuid:
        trace_uuid = await weavel_client.start_trace(user_uuid=user_uuid) # 4. if it is the beginning of the conversation, make a new trace
        
    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )
    
    weavel_client.log.user_message(trace_uuid, messages[-1]["content"]) # 5. Log user message. This will not make extra latency.
    weavel_client.log.assistant_message(trace_uuid, response.choices[0].message.content) # 6. Log assistant message. This will not make extra latency.
    
    return response.choices[0].message.content