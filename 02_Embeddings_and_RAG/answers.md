## ❓Question #1:

The default embedding dimension of `text-embedding-3-small` is 1536, as noted above.

### 1. Is there any way to modify this dimension?

Yes, both for the `text-embedding-3-small` and the `text-embedding-3-large` models, developers can shorten embeddings without the embedding losing its concept-representing properties by passing in the `dimensions` API parameter. So in the case of the `text-embedding-3-small` model, the embedding dimension can be reduced from its maximum 1536 value.

This is specially useful when choosing to work with vector data stores that only support smaller embedding sizes. For example, if the vector data store only supports embeddings up to 1024 dimensions long, developers can still use the `text-embedding-3-small` model and specify a value of 1024 for the `dimensions` API parameter, which will shorten the embedding down from 1536 dimensions, trading off some accuracy in exchange for the smaller vector size.

**References**

[Embedding Models](https://platform.openai.com/docs/guides/embeddings#embedding-models)

[New embedding models and API updates: --Native support for shortening embeddings](https://openai.com/index/new-embedding-models-and-api-updates/)

[OpenAI Create Embeddings API --Dimensions parameter](https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-dimensions)

### 2. What technique does OpenAI use to achieve this?

OpenAI uses a technique called Matryoshka Representation Learning (MRL) to train the newer `text-embedding-3-small` and the `text-embedding-3-large` models which allows developers to trade-off performance and cost. By passing in the `dimensions` API parameter, developers can shorten embeddings (i.e. remove some numbers from the end of the embedding sequence) at the expense of minimally reducing accuracy.

**References**

[Vector embeddings](https://platform.openai.com/docs/guides/embeddings/use-cases)

Paper: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)

## ❓Question #2:

### What are the benefits of using an `async` approach to collecting our embeddings?

The benefits of using an asynchronous client to collecting embeddings are:

- **Higher Throughput and Efficiency**: the async client is able to send multiple embedding requests concurrently.
- **Resource Efficient**: by using an async approach, it is possible to handle many embedding requests with fewer threats or processes, thereby reducing overhead.
- **Better Performance at Scale**: in general, the async client is ideal for batch-processing large amounts of data e.g. batch embedding large corpora.

## ❓ Question #3:

### When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

The Responses API and Chat Completions API are two different ways to interact with OpenAI's models. Although OpenAI's documentation recommends using the Responses API to take advantage of the latest OpenAI platform features, only the Chat Completions API appears to offer a `seed` API parameter that, when specified, will signal to the OpenAI model to make a best effort to sample deterministically such that <u>**repeated requests with the same seed and parameters should return the same result**</u>. However, determinism is not guaranteed, and the documentation recommends checking the `system_fingerprint` response parameter to monitor any changes in the backend: _"[the `system_fingerprint` response parameter] can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism."_

When using the Responses API, more reproducible outputs are more likely to be achieved, though not guaranteed, by:

- Ensuring identical prompt strings
- Setting the `temperature` to 0: This removes randomness and ensures the model chooses the most likely next token at each step.
- ChatGPT suggests fixing the `max_output_tokens`value because _"ensuring consistent output length indirectly contributes to more reproducible behavior"_

**References**
[Open AI API - Create a Model Response](https://platform.openai.com/docs/api-reference/responses/create)
[Open AI API - Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
ChatGPT

## ❓ Question #4:

### What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

By providing the LLM instructions to break down a problem into steps before giving an answer, it is possible to have the LLM return a more thoughtful, detailed response. This can be achieved by adding an instruction to the prompt such as: _"Think through your response step by step."_

### What is that strategy called?

This strategy is called "Chain of Thought" (CoT)
