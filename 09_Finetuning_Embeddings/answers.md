### ❓ Question #1:

#### Describe the nuance between using Q&D pairs to train the embedding model vs. inter-document pairs/related sentences.

Training an embedding model using Q&D pairs vs. inter-document/related sentence pairs leads to different results in retrieval performance.

With Q&D pairs, the model learns to reduce the distance in embedding space between a question and its most relevant document whereas with inter-document/related sentence pairs, the embeddings are supposed to reflect the conceptual similarity between content units without being tied to specific user queries. In the former, retrieval performance is optimized specifically to queries as opposed to the latter, where the focus is on semantic understanding and contextualization. Q&D pairs appears to be more suited for performing direct Q&A tasks while inter-document/related sentence pairs appears to be better for general purpose embedding space where the goal is for similar content to cluster together.

#### What caveats does this approach have? Are there any special considerations for what kind of Q's we should use?

When training using Q&D pairs, it's important to keep in mind that this may lead to:

- **Overspecialization**: If the training data (questions/documents) are from a different domain than the context used in production, the model will likely underperform because the questions won't reference any matching documents.

- **Poor Alignment**: The question/document pairs must truly match semantically; otherwise, misaligned pairs will teach the model incorrect relationships.

- **Chunk Sensitivity**: If the document chunks are too long or too short, it can affect how the model learns what a "relevant" match is.

- **Costly Implementation**: High quality question/document pairs may be time-consuming to generate in specialized domains. Also, due to chunk sensitivity, it may be necessary to spend time experimenting with chunk sizes to balance context and precision.

### 🏗️ Activity #1:

Code for the body of the `create_questions` function:

```
for doc in tqdm.tqdm(documents, desc="Generating questions"):
    # Prepare the input for the chain
    input_context = doc.page_content
    doc_id = doc.metadata["id"]

    # Call the question generation chain
    response = await question_generation_chain.ainvoke({"context": input_context, "n_questions": n_questions})

    # Extract questions
    generated_questions = response.content.split("\n")
    generated_questions = [q.strip() for q in generated_questions if q.strip()]

    # Some outputs might be numbered like "1. What is ...?", so clean numbering
    cleaned_questions = []
    for q in generated_questions:
        if q[0].isdigit() and q[1] == '.':
            cleaned_questions.append(q[2:].strip())
        elif q[0].isdigit() and q[1] == ' ':
            cleaned_questions.append(q[1:].strip())
        else:
            cleaned_questions.append(q)

    # Now save each question
    for q in cleaned_questions:
        question_id = str(uuid.uuid4())
        questions[question_id] = q
        relevant_docs[question_id] = [doc_id]
```

### 🏗️ Activity #2:

#### Both of these losses sound "cool", but what are they - exactly - under the hood?

#### Why are these losses specifically doing? Please write a short summary of each loss.

---

**MultipleNegativesRankingLoss**

This loss function is used to fine-tune models for similarity-based tasks, such as semantic search or information retrieval. It's a highly efficient and effective way to train embeddings when you only have positive pairs (query & relevant document) and no explicit negatives. It works by bringing similar sentence embeddings closer together and pushing dissimilar ones apart; however, it does this without needing explicit negative samples.

_How it works_

Suppose you have a batch of size `N`, where each item is a positive pair: `(anchor_i, positive_i)`.

1. The model computes embeddings for all `2N` sentences.

2. For each anchor `anchor_i`, the positive is `positive_i`.

3. All other positives `{positive_j}`, where `j ≠ i`, are treated as negatives.

4. A similarity score (typically dot-product or cosine similarity) is computed between `anchor_i` and all positives in the batch.

5. The model applies a softmax cross-entropy loss where the correct match is `positive_i`, and all other positives are distractors.

---

**MatryoshkaLoss**

This loss function is designed to train hierarchically nested sentence embeddings where subsets of dimensions in an embedding can still be useful representations. For example, think of the embedding as a full vector (say 384 dimensions). The goal is to ensure that even if you truncate it to just the first 128 or 256 dimensions, it still performs reasonably well. Therefore, similar to Russian Matryoshka dolls, each smaller piece is a functional sub-embedding. The `MatryoshkaLoss` function works with batches of positive pairs (similar to the Multiple Negatives Ranking approach). Bigger batch sizes lead to more effective negatives, thereby improving training quality.

_How it works_

Suppose you have a batch of size `N`, where each item is a positive pair: `(anchor_i, positive_i)`. You define a set of truncation lengths, e.g. [128, 256, 384].

1. The model computes embeddings for both sentences in each pair.

2. For each truncation length `k`, it takes only the first `k` dimensions of the embedding.

3. For each truncated level `k`, compute the cosine similarity between the `anchor[:k]` and `positive[:k]`.

4. It calculates the loss by applying a loss function at each truncation level, then it averages or combines the losses.

---

### ❓Question #2:

Which LCEL RAG Chain do you think answered the questions better, and why?

`finetune_rag_chain` was able to answer all questions whereas `base_rag_chain` did not retrieve relevant context for two of the questions and, as a result of this, it was not able to answer them. Therefore, the RAG chain with the fine-tuned embeddings model proved to do a better job at answering questions from the corpus.

## Task 2: RAGAS Evaluation

It's great to have some idea of how our system is doing based on vibe-checks, but let's use RAGAS to provide more insight info. on how things are improving!

> NOTE: Please recreate _exactly_ the RAGAS process we used to evaluate RAG, baselining with the default retriever, and then comparing the new retriever. The includes the Synthetic Data Generation steps.

```
# IMPORT DEPENDENCIES

from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# CREATE EVALUATOR LLM AND CUSTOM RUN CONFIG FOR EVALUATION
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
custom_run_config = RunConfig(timeout=360, max_wait=180)

#GENERATE SYNTHETIC DATA

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(text_loader.load(), testset_size=10)
dataset.to_pandas()
```

```
# EVALUATE BASELINE RETRIEVER

for test_row in dataset:
  response = base_rag_chain.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)
result
```

```
RESULT | BASELINE RETRIEVER:
{
    'context_recall': 0.3726,
    'faithfulness': 0.7962,
    'factual_correctness(mode=f1)': 0.2914,
    'answer_relevancy': 0.6967,
    'context_entity_recall': 0.2914,
    'noise_sensitivity(mode=relevant)': 0.1412
}
```

```
# EVALUATE FINE-TUNED RETRIEVER

for test_row in dataset:
  response = finetune_rag_chain.invoke({"question" : test_row.eval_sample.user_input})
  test_row.eval_sample.response = response["response"]
  test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]

evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)
result
```

```
RESULT | FINE-TUNED RETRIEVER:
{
    'context_recall': 0.5250,
    'faithfulness': 0.6683,
    'factual_correctness(mode=f1)': 0.3612,
    'answer_relevancy': 0.8606,
    'context_entity_recall': 0.3588,
    'noise_sensitivity(mode=relevant)': 0.2064
}
```
