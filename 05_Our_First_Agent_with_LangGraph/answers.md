### ❓ Question #1:

How does the model determine which tool to use?

The model determines which tool to use through OpenAI's function calling API. When bind_tools() is called, it converts the tools into OpenAI function schemas and adds them to the model's parameters. The model then uses its internal reasoning to analyze the user's input and based on the system prompt and messages, it may decide to call one or several of these tools.

This is handled internally by the model's function calling capability, which has been trained to understand tool descriptions and make appropriate selections based on the context and task at hand.

**References**

[LangChain OpenAI - bind_tools() source code](https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py#L1404)

[OpenAI Docs - Function Calling](https://platform.openai.com/docs/guides/function-calling?api-mode=responses&example=search-knowledge-base#overview)

### ❓ Question #2:

Is there any specific limit to how many times we can cycle?

No, there's no limit to how many times the flow can cycle between the agent and the action nodes.

If not, how could we impose a limit to the number of cycles?

We could impose a limit to the number of cycles by adding a counter in the state that increments each time there's a transition from the `action` node to the `agent` node. Then, we could add another condition to the `should_continue` edge that checks the counter in the state and determines whether the maximum number of cycles has been reached, this would cause the flow to transition directly to the END node.

### 🏗️ Activity #2:

Please write out the steps the agent took to arrive at the correct answer.

1. The state object was populated with the request: "Search Arxiv for the QLoRA paper, then search each of the authors to find out their latest Tweet using Tavily!"
2. The state object was passed to the entry point, which sent the state object to the agent node.
3. The chat model in the agent node determined that the `arxiv` tool needed to be called so an `AIMessage` was added to the messages in the state object.
4. The state object was passed along to the conditional edge.
5. The conditional edge found the "tool_calls" key in `additional_kwarg`, thereby sending the state object to the action node.
6. The action node added the response from the OpenAI function calling endpoint to the messages in the state object and passed the state object along the edge back to the agent node.
7. The chat model in the agent node determined that the `tavily_search_results_json` tool needed to be called so an `AIMessage` was added to the messages in the state object.
8. The state object was passed along to the conditional edge.
9. The conditional edge found the "tool_calls" key in `additional_kwarg`, thereby sending the state object to the action node.
10. The action node added the response from the OpenAI function calling endpoint to the messages in the state object and passed the state object along the edge back to the agent node.
11. The chat model in the agent node determined that no tools needed to be called and that a text response could be sent back to the user. An `AIMessage` with the text response was added to the messages in the state object.
12. The state object was passed along to the conditional edge.
13. The conditional edge received the state object. It could not find the "tool_calls" key in `additional_kwarg` so the state object was passed to the END node and the flow reached its end.

### ❓ Question #3:

How are the correct answers associated with the questions?

The LangSmith client associates the correct answers with the questions in the `create_examples` method by using the **legacy** keyword arguments `inputs` and `outputs`, where `inputs` are the input values for the examples and `outputs` correspond to the output values for the examples. This could be problematic if there's a mismatch between the inputs and the expected outputs. Using the `examples` parameter instead, a dictionary of examples can be used where each example object has its input encapsulated with its output making it easier to scale and less prone to errors.

### ❓ Question #4:

What are some ways you could improve this metric as-is?

The metric currently works by performing case sensitive string comparisons and assigning a boolean score depending on whether the answer provided by the model contains all the required strings. This metric could be improved by: 1) allowing case insensitive string comparisons or performing more sophisticated string comparisons (e.g. based on semantic similarity) 2) assigning a float value as a percentage of the number of required strings found in the provided answer.

### 🏗️ Activity #5:

Please write markdown for the following cells to explain what each is doing.

A new state graph using the StateGraph class is initialized using AgentState as the state schema. Two nodes are then added to the graph. The first node added is labeled `agent` and refers to the `call_model` function.
The second node added is labeled `action` and refers to `tool_node`, which is an instance of the ToolNode class initialized with a specific set of tools.

```
graph_with_helpfulness_check = StateGraph(AgentState)

graph_with_helpfulness_check.add_node("agent", call_model)
graph_with_helpfulness_check.add_node("action", tool_node)
```

Next, the `agent` node is set to be the graph's entry point.

```
graph_with_helpfulness_check.set_entry_point("agent")
```

A function `tool_call_or_helpful` is defined that takes a graph state as the only parameter.
This function appears to define a "smart" edge for a graph. In other words, the function evaluates an initial query with the latest response stored in the graph's state (this corresponds to the last message stored in the graph state's `messages` list). The evaluation is done through a chat model (GPT-4), which will determine if the latest response is helpful or not.

This process is set up using a chain where the initial query and latest response in the graph state are passed as parameters into a prompt template. The resulting prompt is passed as parameter to the chat model and its output is fed into a StrOutputParser(). If the chat model determines that the latest response is helpful in regards to the initial query, then the `tool_call_or_helpful` function will return `end`; otherwise, it will return `continue`. However, it's important to note that the `tool_call_or_helpful` function will short circuit and return `END` if the number of messages in the graph state is greater than 10, in which case the chain will not be run.

```
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def tool_call_or_helpful(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  initial_query = state["messages"][0]
  final_response = state["messages"][-1]

  if len(state["messages"]) > 10:
    return "END"

  prompt_template = """\
  Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

  prompt_template = PromptTemplate.from_template(prompt_template)

  helpfulness_check_model = ChatOpenAI(model="gpt-4")

  helpfulness_chain = prompt_template | helpfulness_check_model | StrOutputParser()

  helpfulness_response = helpfulness_chain.invoke({"initial_query" : initial_query.content, "final_response" : final_response.content})

  if "Y" in helpfulness_response:
    return "end"
  else:
    return "continue"
```

### 🏗️ Activity #4:

Please write what is happening in our `tool_call_or_helpful` function!

Please see the previous answer ☝️

Next, a conditional edge is created using the `tool_call_or_helpful` function which connects the `agent` node with three nodes: 1) itself (the `agent` node) 2) the `action` node, responsible for making tool calls and 3) the END node, where the graph's flow ends.

```
graph_with_helpfulness_check.add_conditional_edges(
    "agent",
    tool_call_or_helpful,
    {
        "continue" : "agent",
        "action" : "action",
        "end" : END
    }
)
```

Next, a normal edge is created from the `action` node to the `agent` node.

```
graph_with_helpfulness_check.add_edge("action", "agent")
```

Next, the state graph is turned into a compiled graph, which implements the Runnable interface.

```
agent_with_helpfulness_check = graph_with_helpfulness_check.compile()
```

Finally, the resulting compiled graph is run asynchronously. Updates returned by the nodes or the tasks are printed as they are completed.

```
inputs = {"messages" : [HumanMessage(content="Related to machine learning, what is LoRA? Also, who is Tim Dettmers? Also, what is Attention?")]}

async for chunk in agent_with_helpfulness_check.astream(inputs, stream_mode="updates"):
    for node, values in chunk.items():
        print(f"Receiving update from node: '{node}'")
        print(values["messages"])
        print("\n\n")
```
