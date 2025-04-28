## Using Ragas to Evaluate a RAG Application built with LangChain and LangGraph

### ❓ Question:

What is the purpose of the `chunk_overlap` parameter in the `RecursiveCharacterTextSplitter`?

The purpose of the `chunk_overlap` parameter is to ensure that information around chunk boundaries is still provided enough context so that it is not mistakenly excluded by the retriever.

### ❓ Question:

Which system performed better, on what metrics, and why?

Results for graph using `retrieve`

```
{
  'context_recall': 0.9292,
  'faithfulness': 0.9128,
  'factual_correctness': 0.6433,
  'answer_relevancy': 0.7800,
  'context_entity_recall': 0.5859,
  'noise_sensitivity_relevant': 0.3334
}
```

Results for graph using `retrieve_adjusted`

```
{
  'context_recall': 0.9396,
  'faithfulness': 0.9516,
  'factual_correctness': 0.6309,
  'answer_relevancy': 0.7819,
  'context_entity_recall': 0.5648,
  'noise_sensitivity_relevant': 0.3082
}
```

The graph using `retrieve_adjusted` performed better on `context_recall`, `faithfulness` and `noise_sensitivity_relevant`.

- `context_recall` ([LLMContextRecall](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.LLMContextRecall)): measures how much of the relevant information from the context is actually used in the generated answer. It makes sense for this metric to improve thanks to contextual compression since less, but more relevant context is supposed to be retrieved and used by the generator pipeline to provide an answer.

- `faithfulness` ([Faithfulness](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.Faithfulness)): measures if the answer sticks to the information passed in the context. It makes sense for this metric to have improved after applying contextual compression because, without contextual compression, the retriever pipeline in the RAG system will pull more noisy or semi-relevant text which increases the probability of generating hallucinations or confusing facts, which results in less faithful answers.

- `noise_sensitivity_relevant` ([NoiseSensitivity](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.NoiseSensitivity)): measures how much the model's answer changes when irrelevant information is added to the context. The lower the noise sensitivity value, the more likely the system's answers will be accurate and focused. With contextual compression, noise in the context is filtered out from the context before it ever reaches the model, which results in more stable and focused answers.

The graph using `retrieve` performed better on `factual_correctness`, `context_entity_recall`.

- `factual_correctness` ([FactualCorrectness](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.FactualCorrectness)): evaluates the final answer against a known correct answer. In this context, it is possible for contextual compression to negatively impact factual correctness if it accidentally removes important facts when retrieving the context. This may cause the final answer to be factually incorrect, even if it's faithful to the context.

- `context_entity_recall` ([ContextEntityRecall](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.ContextEntityRecall)): measures how well the generated answer covers the key entities from the retrieved context. Similar to FactualCorrectness, it is possible for contextual compression to negatively impact context entity recall if it accidentally removes important entities when retrieving the context, in which case they will not be included in the final answer and they will be missing from the full context.

## Using Ragas to Evaluate an Agent Application built with LangChain and LangGraph

### ❓ Question:

Describe in your own words what a "trace" is.

A "trace" is the chronological sequence of events, operations or function calls in a system that make it possible to track its progression over time.

### ❓ Question:

Describe _how_ each of the above metrics are calculated. This will require you to read the documentation for each metric.

- **Tool Call Accuracy**:

  1. Compare model tool calls against ground-truth tool calls
  2. For each tool call, evaluate: tool name (did the model call the correct tool?) and arguments (were the inputs correctly structured and correct in value?)
  3. Compute accuracy, where: _ToolCallAccuracy = Number of Correct Tool Calls / Total Number of Tool Calls_.

- **Agent Goal Accuracy**:

  This is a binary measure indicating whether the AI has achieved the user's goal, assigning a score of 1 for success and 0 for failure. Per [OECD.AI](https://oecd.ai/en/catalogue/metrics/agent-goal-accuracy), the following formula can be derived:

  _AgentGoalAccuracy = Number of Successfully Achieved Goals / Total Number of Goals_.

- **Topic Adherence**:
  Topic Adherence in RAGAS is defined as an F1 score between expected and predicted topics. In ML and information retrieval, the F1 score is a way of measuring how good a prediction is by balancing: **Precision** (how much of what the model said was relevant) and **Recall** (how much of what was relevant, the model actually said). The [formula in the RAGAS documentation](https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/agents.md) is defined as follows:

  _F1 Score = 2 x Precision x Recall / Precision + Recall_
