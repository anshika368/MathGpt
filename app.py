import streamlit as st
import numexpr
import re
import uuid
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import initialize_agent
from langchain_classic.tools import Tool
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.memory import ConversationBufferWindowMemory

##set up the streamlit app
st.set_page_config(page_title="Text to Math Problem Solver", page_icon=":abacus:")
st.title("Text to Math Problem Solver")

# Sidebar configuration
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input(label="Enter your Groq API Key", type="password")

# Session management in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Chat Sessions")

# Initialize session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {st.session_state.session_id: []}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10  # Remember last 10 exchanges
    )

# Display current session ID
st.sidebar.write(f"**Current Session:** `{st.session_state.session_id}`")

# Button to start new chat session
if st.sidebar.button("🆕 New Chat Session"):
    # Save current session
    st.session_state.all_sessions[st.session_state.session_id] = st.session_state.messages.copy()
    # Create new session
    new_session_id = str(uuid.uuid4())[:8]
    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    st.session_state.all_sessions[new_session_id] = []
    st.rerun()

# Show previous sessions
if len(st.session_state.all_sessions) > 1:
    st.sidebar.markdown("**Previous Sessions:**")
    for sess_id in list(st.session_state.all_sessions.keys()):
        if sess_id != st.session_state.session_id:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"📂 {sess_id}", key=f"load_{sess_id}"):
                    # Save current session
                    st.session_state.all_sessions[st.session_state.session_id] = st.session_state.messages.copy()
                    # Load selected session
                    st.session_state.session_id = sess_id
                    st.session_state.messages = st.session_state.all_sessions[sess_id].copy()
                    # Rebuild memory from messages
                    st.session_state.memory = ConversationBufferWindowMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        k=10
                    )
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            st.session_state.memory.chat_memory.add_user_message(msg["content"])
                        else:
                            st.session_state.memory.chat_memory.add_ai_message(msg["content"])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{sess_id}"):
                    del st.session_state.all_sessions[sess_id]
                    st.rerun()

# Clear current chat button
if st.sidebar.button("🗑️ Clear Current Chat"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    st.rerun()

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar to use the app.")    
    st.stop()

language_model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=groq_api_key)

##initialing our tools
wikipedia_wrapper= WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Useful for looking up mathematical concepts, formulas, theorems, or background information on math topics."
)

##initialize the calculator tool for simple arithmetic
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression using numexpr."""
    try:
        # Clean the expression - remove any non-math characters
        cleaned = re.sub(r'[^0-9+\-*/().^%\s]', '', expression)
        cleaned = cleaned.replace('^', '**')  # Convert ^ to Python's power operator
        result = numexpr.evaluate(cleaned).item()
        return f"The numerical result is: {result}"
    except Exception as e:
        return f"Could not evaluate as arithmetic expression. Use the Math Reasoning tool for complex problems."

calculator = Tool(
    name="Calculator",
    func=calculate,
    description="Only for simple arithmetic calculations like 2+2, 10*5, 100/4. Input must be a numerical expression with no variables."
)

##initialize the math reasoning tool for complex problems
math_reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an expert mathematician and teacher. Solve the following math problem with detailed step-by-step explanation.

For the problem, provide:
1. **Understanding the Problem**: Briefly explain what is being asked
2. **Relevant Concepts/Formulas**: List any formulas, rules, or theorems needed
3. **Step-by-Step Solution**: Show each step clearly with explanations
4. **Final Answer**: Clearly state the final answer

Math Problem: {question}

Solution:"""
)

math_reasoning_chain = LLMChain(llm=language_model, prompt=math_reasoning_prompt)

def solve_math_problem(question: str) -> str:
    """Solve complex math problems with step-by-step explanation."""
    try:
        response = math_reasoning_chain.run(question=question)
        return response
    except Exception as e:
        return f"Error solving problem: {str(e)}"

math_solver = Tool(
    name="MathReasoning",
    func=solve_math_problem,
    description="Use this for ANY math problem that requires reasoning, explanation, or involves concepts like: differentiation, integration, calculus, algebra, trigonometry, geometry, equations, word problems, proofs, or anything beyond simple arithmetic. This is the PRIMARY tool for math questions."
)

##initialize the agent with memory for chat history
assistant_agent = initialize_agent(
    tools=[math_solver, calculator, wikipedia_tool],
    llm=language_model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=st.session_state.memory,
    agent_kwargs={
        "prefix": """You are an expert math tutor and problem solver. Your goal is to help students understand and solve any math problem with clear step-by-step explanations.

You have access to the chat history and can refer to previous questions and answers in the conversation.

For ANY math question (calculus, algebra, geometry, trigonometry, word problems, etc.), use the MathReasoning tool to provide detailed solutions.
Only use the Calculator for simple numerical arithmetic like 2+2 or 15*8.
Use Wikipedia only when you need to look up specific mathematical concepts or theorems.

If the user asks a follow-up question about a previous solution, use the context from chat history to provide a relevant answer.
Always aim to educate the user by explaining the reasoning behind each step."""
    }
)

# Display welcome message if no messages
if not st.session_state.messages:
    welcome_msg = """Hello! I am your Math Problem Solver Bot. 🧮

I can help you with:
• **Calculus** - Differentiation, Integration, Limits
• **Algebra** - Equations, Factoring, Polynomials
• **Trigonometry** - Sin, Cos, Tan, Identities
• **Geometry** - Areas, Volumes, Theorems
• **Word Problems** - Step-by-step solutions
• **Simple Calculations** - Arithmetic operations

💡 **Tip:** You can ask follow-up questions! For example:
- "Can you explain step 2 in more detail?"
- "What if the value was different?"
- "Why did you use that formula?"

Ask me any math question and I'll explain the solution step by step!"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Chat input for continuous conversation
if question := st.chat_input("Ask a math question or follow-up..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.spinner("Thinking..."):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response = assistant_agent.run(input=question, callbacks=[st_cb])
        except Exception as e:
            # Fallback: directly use math reasoning for complex errors
            response = solve_math_problem(question)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
        # Save to all_sessions
        st.session_state.all_sessions[st.session_state.session_id] = st.session_state.messages.copy()