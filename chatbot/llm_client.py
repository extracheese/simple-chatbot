import logging
from openai import AzureOpenAI, APIError, Timeout
from typing import List, Dict, Any, Optional, Union
import json
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

# Import configuration safely
try:
    import config
except ImportError:
    print("CRITICAL ERROR: config.py not found or contains import errors.")
    # Handle error appropriately, maybe exit or use defaults
    # For now, set to None to indicate failure
    config = None

# Get logger instance (ensure logging is configured before use)
startup_logger = logging.getLogger('startup')
chat_logger = logging.getLogger('chat')

# Initialize client globally within the module
_azure_client: Optional[AzureOpenAI] = None

# Helper function for formatting messages with detailed logging
def _format_messages_for_api(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Formats a list of LangChain messages or dictionaries into the structure
    required by the Azure OpenAI API, handling potential None content and tool calls.
    """
    chat_logger.debug(f"Starting message formatting for {len(messages)} messages.")
    formatted_messages = []
    last_message_had_tool_calls = False
    for i, msg in enumerate(messages):
        chat_logger.debug(f"Processing message {i+1}/{len(messages)} - Type: {type(msg)}")

        msg_dict = {}
        if isinstance(msg, BaseMessage):
            # Handle LangChain BaseMessage objects
            role = msg.type
            content = msg.content if msg.content else ""

            # Handle AIMessage with tool_calls
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                chat_logger.debug(f"  Formatting AIMessage with tool_calls: {msg.tool_calls}")
                # Ensure content is not None or empty when tool_calls are present
                content = content or ""  # API requires non-null content
                # Convert Langchain tool_calls format if necessary (assuming API needs slightly different structure)
                api_tool_calls = []
                lc_tool_calls = msg.tool_calls if isinstance(msg.tool_calls, list) else [] # Ensure it's a list
                for tc in lc_tool_calls:
                    # Handle potential differences between Langchain and OpenAI tool call formats
                    # This example assumes OpenAI format is expected by the API client
                    if isinstance(tc, dict) and 'id' in tc and 'name' in tc.get('args', {}): # Basic check for Langchain format
                         api_tool_calls.append({
                            "id": tc.get('id'),
                            "type": "function", # Assuming 'function'
                            "function": {
                                "name": tc.get('name'),
                                # Ensure args are JSON strings if needed by API, handle if already string
                                "arguments": json.dumps(tc.get('args', {})) if isinstance(tc.get('args'), dict) else tc.get('args', '{}')
                            }
                         })
                    elif isinstance(tc, dict) and 'id' in tc and 'function' in tc: # Check if already OpenAI format
                         api_tool_calls.append(tc)
                    else:
                         chat_logger.warning(f"Unrecognized tool_call format in AIMessage: {tc}")

                msg_dict = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": api_tool_calls, # Use the potentially converted format
                }
            # Handle ToolMessage
            elif isinstance(msg, ToolMessage):
                chat_logger.debug(f"  Formatting ToolMessage with tool_call_id: {msg.tool_call_id}")
                msg_dict = {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": content, # Tool execution result
                }
            # Handle other BaseMessages (System, Human, standard AI)
            else:
                 chat_logger.debug(f"  Formatting standard BaseMessage (Type: {role})")
                 msg_dict = {"role": role, "content": content}

        elif isinstance(msg, dict):
            # Handle raw dictionaries (potentially from memory)
            role = msg.get("role")
            content = msg.get("content")
            logger = chat_logger
            logger.debug(f"  Formatting dictionary message (Role: {role})")
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Assistant message with tool calls
                    msg_dict = {"role": "assistant", "tool_calls": tool_calls, "content": None}
                    last_message_had_tool_calls = True
                else:
                    # Regular assistant message
                    msg_dict = {"role": "assistant", "content": content or ""} 
                    last_message_had_tool_calls = False
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not last_message_had_tool_calls:
                    logger.warning(f"Tool message {i+1} might not follow an assistant message with tool_calls.")
                msg_dict = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                last_message_had_tool_calls = False # Reset after tool message
            elif role == "user":
                msg_dict = {"role": "user", "content": content}
                last_message_had_tool_calls = False
            elif role == "system":
                msg_dict = {"role": "system", "content": content}
                last_message_had_tool_calls = False
            else:
                logger.warning(f"  Unknown dictionary message role: {role}. Skipping.")
                continue
        else:
            chat_logger.warning(f"  Skipping unknown message type: {type(msg)}")
            continue

        # Ensure content is always a string (important if None was converted)
        if "content" in msg_dict and msg_dict["content"] is None:
             msg_dict["content"] = ""

        # Final check for tool message ordering (heuristic)
        if msg_dict.get("role") == "tool":
            if not formatted_messages or "tool_calls" not in formatted_messages[-1]:
                chat_logger.warning(f"Tool message {i+1} might not follow an assistant message with tool_calls.")

        formatted_messages.append(msg_dict)
        chat_logger.debug(f"  Formatted message: {msg_dict}")

    chat_logger.debug(f"Finished message formatting. Result: {formatted_messages}")
    return formatted_messages

def initialize_client():
    """Initializes the Azure OpenAI client using settings from config.py."""
    global _azure_client
    if not config or _azure_client:
        if _azure_client:
            startup_logger.info("Azure OpenAI client already initialized.")
        else:
            startup_logger.error("Configuration not loaded. Cannot initialize Azure OpenAI client.")
        return

    if not config.AZURE_OPENAI_ENDPOINT or not config.AZURE_OPENAI_API_KEY:
        startup_logger.error("Azure OpenAI Endpoint or API Key missing in configuration.")
        _azure_client = None # Ensure client is None if config is missing
        return

    try:
        _azure_client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_API_VERSION
        )
        startup_logger.info("Azure OpenAI client initialized successfully via llm_client.")
    except Exception as e:
        startup_logger.error(f"Failed to initialize Azure OpenAI client: {e}", exc_info=True)
        _azure_client = None

def get_llm_completion(model: str, messages: List[Union[BaseMessage, Dict[str, Any]]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str = "auto"):
    """Sends a request to the Azure OpenAI API and returns the completion message."""
    if not _azure_client:
        chat_logger.error("LLM client not initialized. Cannot get completion.")
        # Return a simulated error response or raise an exception
        return None # Or raise RuntimeError("LLM client not available")

    # Format messages using the helper function with detailed logging
    formatted_messages = _format_messages_for_api(messages)

    chat_logger.debug(f"Sending request to LLM. Model: {model}, Formatted Messages: {formatted_messages}, Tools: {'Present' if tools else 'Absent'}")

    try:
        response = _azure_client.chat.completions.create(
            model=model,
            messages=formatted_messages, # Pass the API-safe list
            tools=tools if tools else None,  # Use None if no tools
            tool_choice=tool_choice if tool_choice else None, # Use None if no choice
            temperature=0.7, # Adjust creativity/determinism
            max_tokens=800,  # Adjust as needed
            top_p=0.95,      # Adjust as needed
            frequency_penalty=0,
            presence_penalty=0,
            stop=None        # Adjust if specific stop sequences are needed
        )
        chat_logger.debug(f"Raw LLM Response: {response}")

        # Check for refusal or other content filtering issues
        if response.choices and response.choices[0].finish_reason == 'content_filter':
            chat_logger.warning("LLM response terminated due to content filtering.")
            # Return a specific message or handle as needed
            return {"role": "assistant", "content": "I'm sorry, I cannot provide a response due to content restrictions."}

        # Return the first choice's message object (which might contain content or tool_calls)
        if response.choices:
             return response.choices[0].message # This is a ChatCompletionMessage object
        else:
            chat_logger.error("LLM response did not contain any choices.")
            return None

    except APIError as e:
        chat_logger.error(f"Azure OpenAI API error: {e}", exc_info=True)
    except Timeout as e:
        chat_logger.error(f"Azure OpenAI request timed out: {e}", exc_info=True)
    except Exception as e:
        chat_logger.error(f"An unexpected error occurred during LLM request: {e}", exc_info=True)

    return None # Return None or raise error on failure

# Initialize the client when the module is loaded
# Ensure logging and config are set up before this module is imported
initialize_client()
