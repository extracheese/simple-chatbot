import logging
import json
from typing import Dict, Any

# LangChain and OpenAI related imports
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage, FunctionMessage

# Import custom modules and components
from .order import Order
from .state_manager import get_or_create_memory, update_memory
from .llm_client import get_llm_completion
from .tools import search_menu, update_order_tool, confirm_order, tools_schema
from menu.processor import MenuProcessor
import config # Import configuration

# Get logger instances
chat_logger = logging.getLogger('chat')

# === Main Chat Logic === #

def generate_response(session_id: str, user_message: str, current_order: Order, processor: MenuProcessor):
    """Generates a response using the LLM, handling context, history, and function calls.
    
    Args:
        session_id: Unique identifier for the user session.
        user_message: The message received from the user.
        current_order: The current state of the user's order.
        processor: The MenuProcessor instance.
        
    Returns:
        A dictionary containing the response, session_id, and updated order state.
        Returns None if a critical error occurs.
    """
    chat_logger.info(f"Session {session_id} - Orchestrator received message: '{user_message}'")

    # Determine model to use for this request
    model_choice = config.AZURE_OPENAI_MODEL

    # 1. Retrieve memory for the session
    memory: ConversationBufferMemory = get_or_create_memory(session_id)

    # 2. Construct the prompt with history and current order state
    system_prompt = f"""You are a helpful McDonald's order assistant. 
    Your goal is to help the user build their order and confirm it. 
    Be friendly and efficient. Ask clarifying questions if the user's request is ambiguous. 
    Do not make up items not on the menu. 
    Use the available tools to search the menu, update the order, or confirm it.
    Current Order State: {current_order.to_dict()}
    """
    
    # Convert LangChain memory messages to the format expected by OpenAI API
    history_messages = memory.chat_memory.messages
    llm_messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=user_message)]
    # Convert to dict format
    llm_messages = [msg.dict() for msg in llm_messages]
    # Remove 'type' key if present, as OpenAI uses 'role'
    for msg_dict in llm_messages:
        if 'type' in msg_dict:
            # Map Langchain type to OpenAI role
            role = msg_dict.pop('type')
            if role == 'human':
                msg_dict['role'] = 'user'
            elif role == 'ai':
                msg_dict['role'] = 'assistant'
            elif role == 'system':
                 msg_dict['role'] = 'system'
            elif role == 'function': # Handle function/tool call messages
                 msg_dict['role'] = 'tool'
                 # Ensure 'content' is present, even if None (OpenAI API requirement)
                 msg_dict['content'] = msg_dict.get('content') 
                 # Keep 'tool_call_id' and 'name' as needed
            else:
                 msg_dict['role'] = role # Keep other potential roles
        # Ensure 'content' exists
        if 'content' not in msg_dict:
             msg_dict['content'] = None # Add None if missing

    # 3. Make the first LLM call (potentially requesting tool use)
    chat_logger.debug(f"Session {session_id} - Sending first request to LLM. Messages: {llm_messages}")
    response_message = get_llm_completion(
        model=model_choice,
        messages=llm_messages,
        tools=tools_schema, # Provide tool definitions
        tool_choice="auto"  # Let the LLM decide whether to use tools
    )

    if not response_message:
        chat_logger.error(f"Session {session_id} - Failed to get a valid response from LLM on first call.")
        # Return a user-facing error message
        return {
            "response": "I'm sorry, I encountered an issue processing your request. Please try again.",
            "session_id": session_id,
            "order": current_order.to_dict() # Return current known order state
        }

    # 4. Check if the LLM wants to call a function/tool
    if response_message.tool_calls:
        chat_logger.info(f"Session {session_id} - LLM requested tool call(s): {response_message.tool_calls}")
        # For simplicity, handle the first tool call requested
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = {}
        try:
            function_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
             chat_logger.error(f"Session {session_id} - Failed to parse JSON arguments for {function_name}: {tool_call.function.arguments}")
             # Handle error: maybe ask user to rephrase or return error message
             function_response_string = f"Error: Invalid arguments provided for {function_name}."
             tool_message = FunctionMessage(name=function_name, content=function_response_string, tool_call_id=tool_call.id)
        else:
            function_response_string = None # To store the string result for logging/memory

            # Execute the corresponding function from tools.py
            try:
                if function_name == "search_menu":
                    result_string, _ = search_menu(
                        processor=processor, 
                        query=function_args.get('query', ''), 
                        current_order=current_order
                    )
                    function_response_string = result_string
                elif function_name == "update_order_tool":
                    function_response_string = update_order_tool(
                        session_id=session_id,
                        order_updates=function_args,
                        current_order=current_order,
                        processor=processor
                    )
                elif function_name == "confirm_order":
                    function_response_string = confirm_order(
                        session_id=session_id,
                        current_order=current_order
                    )
                else:
                    chat_logger.warning(f"Session {session_id} - Unknown function call requested: {function_name}")
                    function_response_string = f"Unknown function: {function_name}"
            except Exception as e:
                 chat_logger.error(f"Session {session_id} - Error executing tool {function_name}: {e}", exc_info=True)
                 function_response_string = f"An error occurred while trying to {function_name}."

            # Ensure the response is a string for the FunctionMessage
            if not isinstance(function_response_string, str):
                 chat_logger.error(f"Session {session_id} - Tool {function_name} did not return a string. Got: {type(function_response_string)}")
                 function_response_string = "Error executing tool internally."

            # Prepare the FunctionMessage for the next LLM call
            tool_message = FunctionMessage(
                name=function_name, 
                content=function_response_string, 
                tool_call_id=tool_call.id
            )

        chat_logger.info(f"Session {session_id} - Function {function_name} executed. Result preview: {function_response_string[:100]}...")

        # 5. Add tool results to the history and make the second LLM call
        # Convert response_message and tool_message back to LangChain format for memory
        formatted_tool_calls = []
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                if tc.type == 'function': # Ensure it's a function call
                    formatted_tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments) # Parse JSON string args
                            # LangChain AIMessage expects 'args' as dict, not string
                        }
                    )
                else:
                     chat_logger.warning(f"Session {session_id} - Skipping non-function tool call type: {tc.type}")
        
        # Construct AIMessage with potentially formatted tool_calls
        ai_message_to_add = AIMessage(
            content=response_message.content if response_message.content else "", # Ensure content is not None
            # Pass the formatted list of dicts, or None if empty
            tool_calls=formatted_tool_calls if formatted_tool_calls else None 
        )
        memory.chat_memory.add_message(ai_message_to_add)
        chat_logger.debug(f"Session {session_id} - Added AIMessage to memory: {ai_message_to_add}")

        memory.chat_memory.add_message(tool_message)
        update_memory(session_id, memory) # Save updated memory
        
        # Prepare messages for the second LLM call (OpenAI format)
        llm_messages.append(response_message.dict()) # Add original AI response (dict)
        # Convert LangChain FunctionMessage back to OpenAI 'tool' role format
        tool_msg_dict = tool_message.dict()
        tool_msg_dict['role'] = 'tool'
        tool_msg_dict.pop('type', None)
        tool_msg_dict['tool_call_id'] = tool_message.tool_call_id # Ensure tool_call_id is present
        llm_messages.append(tool_msg_dict) # Add tool execution result (dict)

        chat_logger.debug(f"Session {session_id} - Sending second request to LLM with tool results. Messages: {llm_messages}")
        second_response_message = get_llm_completion(
            model=model_choice,
            messages=llm_messages,
            tools=tools_schema, # Provide tool definitions
            tool_choice="none"  # Prevent tool usage in the second call
        )

        if second_response_message and second_response_message.content:
            final_response_text = second_response_message.content
            # Add final AI response to memory
            memory.chat_memory.add_ai_message(final_response_text)
        else:
            chat_logger.error(f"Session {session_id} - Failed to get valid content from second LLM response.")
            final_response_text = "I seem to have lost my train of thought after using my tools. Could you please repeat your request?"
            # Add this error response to memory as AI message
            memory.chat_memory.add_ai_message(final_response_text)
            
        # Update memory after the second call's response
        update_memory(session_id, memory)

    else:
        # 6. If no tool call was requested, use the response directly
        final_response_text = response_message.content
        if not final_response_text:
             chat_logger.warning(f"Session {session_id} - LLM response had no content and no tool calls.")
             final_response_text = "I'm not sure how to respond to that. Can you try rephrasing?"
        
        # Add user message and AI response to memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(final_response_text)
        update_memory(session_id, memory) # Save updated memory

    chat_logger.info(f"Session {session_id} - Generated final response: '{final_response_text[:100]}...' Order state: {current_order.status.value}")

    # 7. Return the final response and state
    return {
        "response": final_response_text,
        "session_id": session_id,
        "order": current_order.to_dict() # Return the latest state of the order
    }
