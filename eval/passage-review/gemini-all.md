You are reviewing test questions for a search system. Each item has a QUESTION and a QUOTE from a document.
The search system must find the document passage containing the QUOTE using only the QUESTION.
For each item answer PASS or FAIL, with one short reason if FAIL. FAIL if any of these is true:
1. The quote does not actually answer the question.
2. The question cannot be understood without seeing the document (it says "the example", "the tutorial" with no way to tell which, etc.) so that many other passages would answer it equally well.
3. The question gives away the answer or copies long wording from the quote.
4. The question is wrong, confusing, or asks two unrelated things.
Do not reword questions. Reply as a list: number, PASS or FAIL, reason.

## 1
QUESTION: What happens to an ad whose targeting data has no minimum age, if it is for something like gambling?
QUOTE: 4.1 If targeting metadata leaves minimum age unset, treat the ad as reaching all ages: restricted categories (alcohol, gambling, weight loss, financial services) are then rejected regardless of creative content.

## 2
QUESTION: How is a product listing treated if it names a high-end label, is cheap, and claims to be the real thing?
QUOTE: 3.1 A listing that names a luxury brand and is priced under $100 is flagged for counterfeit review, and if it also asserts authenticity it is rejected instead.

## 3
QUESTION: How long and how expensive is a normal notebook run of the moderation example, and which switch turns on the whole table?
QUOTE: The notebook's own evaluation cell runs 2 samples per domain by default to keep it cheap;
set `RUN_FULL_EVAL=1` for the full table. A default notebook run takes about 10 to 15
minutes and costs on the order of a dollar.

## 4
QUESTION: Apart from the two main factorial designs, how many extra targeted studies did the authors run, and how many distinct configurations did these cover in total?
QUOTE: The headline analyses in the main text rest on the 167-configuration D-optimal fractional factorial (retrieval ANOVA) and the 122-configuration faithfulness factorial. Beyond that core design, we ran 24 targeted ablation and model-comparison experiments totalling 138 distinct configurations, plus the 5K–200K corpus-scaling sweep (18 cells \times 1–3 seeds = 72 runs; see Sections 5 and B.3.9).

## 5
QUESTION: How much smaller was the real benefit of stacking the best three retrieval methods than the total expected from using each alone?
QUOTE: How far a combined pipeline’s gain falls below the gain predicted by adding up its techniques measured one at a time; 63\% for the three strongest retrieval techniques (Section 5).

## 6
QUESTION: Why were seven of the supported benchmarks left out of the evaluation, and which one was named as too large?
QUOTE: The remaining 7 supported benchmarks (MS MARCO, TREC-COVID, TriviaQA, Natural Questions, FEVER, 2WikiMultiHopQA, MultiHop-RAG) were not evaluated due to infrastructure constraints: corpus sizes exceeding our Milvus Lite capacity (MS MARCO: 8.8M passages), rate-limited LLM proxy access for generation evaluation, and time constraints. All adapters are functional and available for community use.

## 7
QUESTION: Which model judged faithfulness throughout the study, and with what sampling settings?
QUOTE: All faithfulness evaluations (both the factorial ANOVA and the profile comparisons) use the same GPT-5.1 judge model with identical RAGAS v0.4 configuration (T{=}0, deterministic seed, batch_size{}=4).

## 8
QUESTION: In the executed subset, how many of the main effects and two-way interactions can be estimated, and what is the range of cell sizes?
QUOTE: The 167 executed configurations cover all factor levels in 7 of 8 dimensions and 3 of 4 levels in the enrichment dimension (D3). All 22 main effects and all 28 two-way interactions are estimable from this subset. However, the design is unbalanced (cell sizes range from 1 to 7), which affects the precision of individual interaction estimates.

## 9
QUESTION: In the example of generating code from comments, which statement was missing from the model's output?
QUOTE: The generated code looks like working code. However, the output doesn't contain an important `import` statement, i.e., `import json`. While these models generate working code, you must pay close attention to small details like this and always make sure to test your code.

## 10
QUESTION: What does Toggle do during budget-limited training phases in Kimi K2.5?
QUOTE: Kimi K2.5 introduces Toggle, a training heuristic that alternates between inference-time scaling and budget-constrained optimization. During budget-limited phases, the model is trained to solve problems within a task-dependent token budget, producing more concise chain-of-thought reasoning.

## 11
QUESTION: Why are tool definitions seen as the most important part of function calling?
QUOTE: Tool definitions are arguably the most critical component of function calling. They are the only way the LLM knows what tools are available and when to use them.

## 12
QUESTION: What lesson do the authors draw about relying on compact models to produce reasoning explanations?
QUOTE: This approach shows a clever use of small language models for rationale generation. The results are remarkable given that larger language models are preferred for this capability over smaller ones. Decomposing tasks in this way is something developers should think deeply about. Not everything needs to be done by the large models.

## 13
QUESTION: What can be done when an LLM does not model a needed role well?
QUOTE: - **Role-playing capability**: LLM-based agents typically need to adapt a role to effectively complete tasks in a domain. For roles that the LLM doesn't characterize well, it's possible to fine-tune the LLM on data that represent uncommon roles or psychology characters.

## 14
QUESTION: What two abilities let agents handle complex tasks and be extended with external tools?
QUOTE: LLM-based agents rely on two key capabilities to solve complex tasks: tool calling and reasoning. These capabilities allow agents to be augmented with external tools, connect to MCP (Model Context Protocol) servers, and access knowledge bases.

## 15
QUESTION: How do deep agents keep working toward goals across long periods instead of improvising step by step?
QUOTE: Instead of reasoning ad-hoc inside a single context window, Deep Agents maintain structured task plans they can update, retry, and recover from. Think of it as a living to-do list that guides the agent toward its long-term goal. To experience this, just try out Claude Code or Codex for planning; the results are significantly better once you enable it before executing any task.

## 16
QUESTION: Why should settings be read inside forward and not in the constructor of a module?
QUOTE: **`Module.forward(**kwargs)`**
What you override. Accept signature inputs as keyword arguments, do your work, return a `Prediction`. Don’t put settings reads in `__init__` — read them inside `forward` so the current LM and adapter are picked up at call time, not at construction time.

## 17
QUESTION: What does the public call of an RLM do before it starts its loop, and when does the loop stop?
QUOTE: **`__call__(**inputs, interpreter_factory=...)` / `acall(**inputs, interpreter_factory=...)`**
The public call validates the inputs against the signature, builds the variable list, opens the interpreter, and runs the loop. Each turn asks the action predictor for code, runs it, and appends the result to history until the model submits or the loop reaches `max_iters`.

## 18
QUESTION: When is token-usage information available on a prediction?
QUOTE: **`Prediction.get_lm_usage()` / `Prediction.set_lm_usage(tokens_dict)`**
Token-usage accounting. Populated when the prediction ran under `dspy.settings(track_usage=True)`.

## 19
QUESTION: What do the optimizers actually change in a program to improve a metric?
QUOTE: **What do DSPy optimizers tune?** Or, _what does compiling actually do?_ Each optimizer is different, but they all seek to maximize a metric on your program by updating prompts or LM weights.

## 20
QUESTION: Which adapter suits a model that formats its answers unreliably?
QUOTE: Same Signature, different prompt shape, no Signature changes. LMs vary a lot in what they prefer to read and produce: a model trained on instruction-following with markers loves ChatAdapter’s `[[ ## field ## ]]` format; a model with native structured-output mode does best with JSONAdapter; a reasoning model that formats unreliably wants TwoStepAdapter.

## 21
QUESTION: At what point does KNNFewShot choose its demonstrations, and what does it do with the training set when it is built?
QUOTE: The trainset is embedded once at construction. Each forward call embeds the new input, retrieves the `k` nearest training examples, and runs `BootstrapFewShot` on that micro-trainset before invoking the program.

## 22
QUESTION: What risk is shown when a model that includes a password is used for both input and output?
QUOTE: Now, whenever a browser is creating a user with a password, the API will return the same password in the response.

## 23
QUESTION: What advantage does sending a series of items one by one give the receiving side?
QUOTE: So, it will send the first item, the client will receive and start processing it, and you might still be producing the next item.

## 24
QUESTION: In the tutorial on returning several model types, why does putting two model classes joined by a vertical bar directly into the response model argument raise an error?
QUOTE: But if we put that in the assignment `response_model=PlaneItem | CarItem` we would get an error, because Python would try to perform an **invalid operation** between `PlaneItem` and `CarItem` instead of interpreting that as a type annotation.

## 25
QUESTION: If the other side drops the connection in a FastAPI websocket handler, what is raised, and what can the developer do with it?
QUOTE: When a client disconnects, a `WebSocketDisconnect` exception is raised, you can catch it.

## 26
QUESTION: When a path operation returns a Response object directly, what conversion work does FastAPI skip?
QUOTE: It won't do any data conversion with Pydantic models, it won't convert the contents to any type, etc.

## 27
QUESTION: For the FastEmbed ranker, which two settings control where the model files live and how much parallelism the runtime gets?
QUOTE: You can set the path where the model is stored in a cache directory. You can also set the number of threads a single `onnxruntime` session can use.

## 28
QUESTION: Where is the distance measure for the Oracle embedding retriever chosen, and besides the query vector which two arguments does the retriever accept?
QUOTE: The distance metric (COSINE, EUCLIDEAN, or DOT) is configured on the `OracleDocumentStore`. In addition to `query_embedding`, the retriever accepts `top_k` (maximum documents to return) and `filters` to narrow the search space.

## 29
QUESTION: What does the component that merges branches do with several values that arrive, and what is it useful for?
QUOTE: `BranchJoiner` receives multiple data connections of the same type from other components and passes the first value it receives to its single output. This makes it essential for closing loops in pipelines or reconciling multiple branches from a decision component.

## 30
QUESTION: How can you get at the saved pipeline state after a run crashes?
QUOTE: Wrap `pipeline.run()` in a `try`/`except` block and retrieve the snapshot from the raised `PipelineRuntimeError`:

## 31
QUESTION: Which model does the NVIDIA ranker use if none is given?
QUOTE: `NvidiaRanker` ranks `Documents` based on semantic relevance to a specified query. It uses ranking models provided by [NVIDIA NIMs](https://ai.nvidia.com). If you don't set the `model` parameter, the hosted default `nv-rerank-qa-mistral-4b:1` is used.

## 32
QUESTION: Which optional search settings can be passed to the MongoDB Atlas full-text retriever when it runs?
QUOTE: When running the component, you can specify more optional parameters such as `fuzzy` or `synonyms`, `match_criteria`, `score`. Check out our [MongoDB Atlas](/reference/integrations-mongodb-atlas) API Reference for more details on all parameters.

## 33
QUESTION: What two things must a Vespa document store provide for embedding retrieval, and what are their default names?
QUOTE: - A tensor field for embeddings (named `embedding` by default, configurable on the Document Store via `embedding_field`).
- A rank profile that scores nearest-neighbor candidates (named `semantic` by default, configurable via the `ranking` parameter).

## 34
QUESTION: Who is responsible for cleaning user-supplied URLs and file paths before they reach Haystack?
QUOTE: Haystack is a framework intended to run inside a trusted execution environment. It assumes that the application built with it has already validated and sanitized user-supplied input before passing it to the framework. Validation and sanitization of input, for example URLs, file paths, filter expressions, and queries, are the responsibility of the application, not Haystack.

## 35
QUESTION: What must be provided to train a cross-encoder with the trainer, and which items are optional?
QUOTE: ```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training.

## 36
QUESTION: Besides the evaluation loss, how can you get richer measurements while training a cross-encoder?
QUOTE: ```{eval-rst}
You can provide the :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` with an ``eval_dataset`` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training.

## 37
QUESTION: Why is learning from several datasets together usually hard, and what lets the cross-encoder trainer handle it?
QUOTE: ```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets.

## 38
QUESTION: What do the example training scripts for sentence embedding models fine-tune?
QUOTE: The [training](training/) folder contains examples how to fine-tune transformer models like BERT, RoBERTa, or XLM-RoBERTa for generating sentence embedding. For the documentation how to train your own models, see [Training Overview](../../docs/sentence_transformer/training_overview.md).

## 39
QUESTION: Which function quantizes OpenVINO models to 8-bit, and where does it save the result?
QUOTE: OpenVINO models can be quantized to int8 precision using Optimum Intel to speed up inference.
To do this, you can use the `export_static_quantized_openvino_model` function,
which saves the quantized model in a directory or model repository that you specify.
Post-Training Static Quantization expects:

## 40
QUESTION: Where do extra arguments given as model kwargs end up for ONNX models?
QUOTE: All keyword arguments passed via `model_kwargs` will be passed on to `ORTModel.from_pretrained`. Some notable arguments include:

## 41
QUESTION: How do you pick and order which Intel GPUs a training run uses?
QUOTE: ```cli
ZE_AFFINITY_MASK=2,0 torchrun trainer-program.py ...
```

## 42
QUESTION: Which cache class do models use by default, and how does its size behave as text is produced?
QUOTE: The [`DynamicCache`] is the default cache class for all models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.

## 43
QUESTION: What two settings must a custom model implement so that attention backends can be swapped?
QUOTE: Use `ALL_ATTENTION_FUNCTIONS` when defining the attention layer and propagate `**kwargs` from the base `MyModel` class to the attention layers. Set `_supports_attention_backend` to `True` in [`PreTrainedModel`].

## 44
QUESTION: How many weights does AQLM treat as one group, and what do they get expressed as?
QUOTE: Additive Quantization of Language Models ([AQLM](https://huggingface.co/papers/2401.06118)) quantizes multiple weights together and takes advantage of interdependencies between them. AQLM represents groups of 8-16 weights as a sum of multiple vector codes.

## 45
QUESTION: Which image processor backend is the default, and why is it preferred?
QUOTE: - [`BaseImageProcessor`] — abstract base class (for backward compatibility only; do not instantiate directly)
  - [`TorchvisionBackend`] — the **default** [torchvision-backed](https://pytorch.org/vision/stable/index.html) backend. GPU-accelerated and significantly faster than the PIL backend. All models expose a `<Model>ImageProcessor` class that inherits from it.

## 46
QUESTION: How many threads load parameters asynchronously by default, and why was that number chosen?
QUOTE: > [!NOTE]
> The default is 4 threads for asynchronous parameter loading. This provides the best trade-off across loading scenarios and hardware. The work is mostly I/O bound, but depending on accelerator hardware and the `dtype` required at loading, it can become CPU/GPU-bound if the `dtype` differs from the serialized one (this requires an additional copy operation).

## 47
QUESTION: Which precision flag should you use on older GPUs rather than Ampere ones, and what does gradient accumulation simulate?
QUOTE: - Set `bf16=True` for fast mixed precision training if your hardware supports it (Ampere+ GPUs). Otherwise, fall back to `fp16=True` on older hardware.
- `gradient_accumulation_steps` simulates a larger effective batch size by accumulating gradients over multiple forward passes before updating weights.

## 48
QUESTION: How do you turn on layer-wise variants of an optimizer, and which layers get low-rank treatment?
QUOTE: Append `_layerwise` to the optimizer name for layerwise optimization (`"galore_adamw_layerwise"`). Only linear layers targeted by GaLore use low-rank decomposition. All other layers are optimized normally.

## 49
QUESTION: Why are models with custom code skipped when looking up a conversion mapping?
QUOTE: Custom code models are skipped before any lookup happens. A sub-model counts as custom code when [`~PreTrainedModel.is_custom_code`] returns `True`, which covers models loaded from the Hub with `trust_remote_code=True` and any model class not in Transformers. Custom code models don't inherit a conversion mapping because its class name or `model_type` might collide with a native model.

## 50
QUESTION: How can you inspect the text an LLM will actually read for a document?
QUOTE: Then, we can test what the LLM will actually end up reading using the `get_content()` function and specifying `MetadataMode.LLM`:

## 51
QUESTION: In what order does FastAPI run the cleanup code of dependencies that use yield when they are nested?
QUOTE: **FastAPI** will make sure that the "exit code" in each dependency with `yield` is run in the correct order.

## 52
QUESTION: Without a declared output model, what does FastAPI do to returned data before sending it?
QUOTE: If you don't declare a response model, FastAPI will use the `jsonable_encoder` explained in [JSON Compatible Encoder](../tutorial/encoder.md) and put it in a `JSONResponse`.

## 53
QUESTION: What three things does a new agent need when it is created in the agent tutorial?
QUOTE: Now we create our agent. It needs an array of tools, an LLM, and a system prompt to tell it what kind of agent to be. Your system prompt would usually be more detailed than this!

## 54
QUESTION: Which callback counts tokens for cost analysis?
QUOTE: Tokens are counted using the `TokenCountingHandler` callback. See the [example notebook](/python/examples/observability/tokencountinghandler) for details on the setup.

## 55
QUESTION: How can you produce a retrieval test set without writing every question by hand?
QUOTE: You can manually curate a retrieval evaluation dataset of questions + node id's. We also offer synthetic dataset generation over an existing text corpus with our `generate_question_context_pairs` function:

## 56
QUESTION: Which ports can the bidirectional connection protocol in the full-stack tutorial work over?
QUOTE: WebSockets are a communication protocol that enables bidirectional and full-duplex communication between a client and a
server over a single, long-lived connection. The WebSocket protocol is designed to work over the same ports as HTTP and
HTTPS (ports 80 and 443, respectively) and uses a similar handshake process to establish a connection.

## 57
QUESTION: What happens to a document's attributes when a node parser breaks it into chunks?
QUOTE: Node parsers are a simple abstraction that take a list of documents, and chunk them into `Node` objects, such that each node is a specific chunk of the parent document. When a document is broken into nodes, all of it's attributes are inherited to the children nodes (i.e. `metadata`, text and metadata templates, etc.).

## 58
QUESTION: What naming pattern do packages for data connectors follow on the Python package index?
QUOTE: - Browse the [`readers` directory](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers) of the LlamaIndex repo. Each package has a README with installation and usage instructions.
- Search PyPI for `llama-index-readers-`. Package names follow the pattern `llama-index-readers-<source>`, for example `llama-index-readers-google` or `llama-index-readers-database`.

## 59
QUESTION: What minimum version of the Python SDK is needed for the client tutorial?
QUOTE: - Mac or Windows computer
- Latest Python version installed
- Latest version of `uv` installed
- You must use the Python MCP SDK 2.0.0 or higher

## 60
QUESTION: Can a server give a freshness lifetime without announcing list-change notifications, and what then happens on the client?
QUOTE: - A server **MAY** provide `ttlMs` without advertising `listChanged: true` in its
  capabilities. In this case, the client relies entirely on TTL-based freshness.
- A server **MAY** advertise `listChanged: true` **and** provide `ttlMs`. In this case,
  the client can use the TTL to avoid unnecessary refetches between notifications, and
  the notification acts as an immediate invalidation signal.

## 61
QUESTION: How can servers describe folders among the file:// items, given they have no standard media type?
QUOTE: MCP servers **MAY** identify file:// resources with an
[XDG MIME type](https://specifications.freedesktop.org/shared-mime-info-spec/0.14/ar01s02.html#id-1.3.14),
like `inode/directory`, to represent non-regular files (such as directories) that don’t
otherwise have a standard MIME type.

## 62
QUESTION: Which two fields handle version negotiation in the discovery exchange?
QUOTE: 1. **Protocol Version Selection**: The `io.modelcontextprotocol/protocolVersion` field declares the version the client is speaking on this request, and `supportedVersions` in the response lists the versions the server accepts.

## 63
QUESTION: What does giving arguments as one JSON object do to a ZIP code value in the inspector command line?
QUOTE: `--tool-args-json` takes the whole argument object at once and skips the `key=value` parsing, so `{"zip":"10001"}` sends the string `"10001"` rather than a number. The schema conversion still applies to string values, so this only keeps a string a string when the schema types that property as a string (or doesn't declare it). The two flags are mutually exclusive:

## 64
QUESTION: What do roots tell servers, and how does a server learn about changes to them?
QUOTE: Roots are exclusively filesystem paths and always use the `file://` URI scheme. They help servers understand project boundaries, workspace organization, and accessible directories. The roots list can change as users work with different projects or folders. Servers pick up the updated boundaries the next time they request the roots list.

## 65
QUESTION: How many clients does the reference debugging tool provide, behind a single program?
QUOTE: The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the reference developer tool for testing and debugging [MCP servers](/docs/2026-07-28/learn/server-concepts). It ships as a single package, `@modelcontextprotocol/inspector`, providing **three clients behind one binary**:

## 66
QUESTION: What should an agent do when its conversation grows too long, according to the fix for context distraction?
QUOTE: **Solution:** After a certain number of turns or when the context grows too large, the agent should **summarize the most recent and relevant parts of the conversation** – focusing on your current travel dates and destination – and use that condensed summary for the next LLM call, discarding the less relevant historical chat.

## 67
QUESTION: When is it worth using several agents instead of one, and when is one enough?
QUOTE: **Answer: A2.** Multiple agents are useful when a task has distinct specialist roles, separate tools or permissions, and clear handoffs that need coordination. A simple support bot can usually remain a single agent. A high volume of identical requests is primarily a scaling concern and does not by itself require a multi-agent design.

## 68
QUESTION: Why does tampering with a signed payload make verification fail?
QUOTE: Verification fails. The signature was computed over the canonical bytes of the original payload; modifying any field changes those bytes, which makes the signature invalid. The attacker would need the private key to produce a fresh valid signature, which they do not have.

## 69
QUESTION: How does managing information across a conversation differ from writing one static set of instructions?
QUOTE: Prompt engineering is focused on a single set of static instructions to effectively guide the AI Agents with a set of rules.  Context engineering is how to manage a dynamic set of information, including the initial prompt, to ensure that the AI Agent has what it needs over time. The main idea around context engineering is to make this process repeatable and reliable.

## 70
QUESTION: How should an agent's access to important systems be limited?
QUOTE: **Mitigation**: AI agents should have access to systems on a need-only basis to prevent these types of attacks. Communication between the agent and system should also be secure. Implementing authentication and access control is another way to protect this information.

## 71
QUESTION: How does remembered information about a traveller shape later trip planning?
QUOTE: A long-term memory might store that "Ben enjoys skiing and outdoor activities, likes coffee with a mountain view, and wants to avoid advanced ski slopes due to a past injury". This information, learned from previous interactions, influences recommendations in future travel planning sessions, making them highly personalized.

## 72
QUESTION: Why do human reviewers still matter when evaluating voice agents?
QUOTE: Human reviewers can fill gaps in automated evaluation, particularly for subjective audio qualities such as naturalness, prosody, pronunciation, and whether the conversation feels appropriately paced.

## 73
QUESTION: Why are two separate OpenAI keys needed for the Databricks sandbox example, and what must they share?
QUOTE: - An OpenAI application key (`OPENAI_API_KEY`) for creating sessions and submitting input.
- A separate OpenAI executor key (`OPENAI_EXECUTOR_API_KEY`) with the same organization, project, and user or service account as the session. It is passed to codex in the sandbox as `CODEX_API_KEY`.

## 74
QUESTION: What does the adapter do in the AWS flight demo?
QUOTE: - **Model Context Protocol (MCP)** lets ChatGPT discover tools, call them with
  structured inputs, and receive structured results. The MCP adapter is the
  server that exposes the three flight tools and validates their inputs and
  outputs against schemas, which define the accepted fields and values.
- **The widget** is a small HTML view inside ChatGPT.

## 75
QUESTION: When should the portfolio manager agent call a single specialist on its own?
QUOTE: **2. For ad-hoc or follow-up analysis (e.g., user requests only one section, or you need to re-run a single specialist):**
- Use the relevant individual specialist tool.

## 76
QUESTION: Why is a pleasant-sounding spoken confirmation not enough to show a voice agent did its job?
QUOTE: A complete evaluation follows a request through the entire system: the user’s spoken intent, the agent’s conversational behavior, the work delegated, the tools actually executed, the resulting application state, and the result ultimately communicated to the user. A natural-sounding confirmation does not prove the correct backend actions were completed.

## 77
QUESTION: Building reliable systems out of unreliable parts is compared to which field?
QUOTE: This paradigm of trying to build a reliable system out of less reliable components is reminiscent of probabilistic programming, and many of the analysis techniques of that field can be applied to this one.

## 78
QUESTION: Does the SharePoint URL inspection command change the ChatGPT allowlist?
QUOTE: This command calls Microsoft Graph and prints each resolved site identifier and
the deduplicated collection GUIDs. It does not read or modify the ChatGPT
workspace allowlist. Tenant-root URLs and percent-encoded paths, such as
`https://contoso.sharepoint.com/sites/Finance%20Team`, are both supported.

## 79
QUESTION: What does the gating network in a mixture of experts do, and what ensures the weights sum to one?
QUOTE: - **Gating Network:** The gating network is responsible for dynamically selecting which experts should be consulted for a given input. It assigns weights to each expert's output, determining their contribution to the final prediction. The gating network typically uses a softmax function to produce these weights, ensuring they sum to one.

## 80
QUESTION: In the conference-room translation demo, how do listeners get the speaker's words in their own language?
QUOTE: A real-world use case for this demo is a multilingual, conversational translation where a speaker talks into the speaker app and listeners hear translations in their selected native language via the listener app. Imagine a conference room with a speaker talking in English and a participant with headphones in choosing to listen to a Tagalog translation.

## 81
QUESTION: Which version of the tracing API should instrumented packages depend on, and when should they update?
QUOTE: - OpenTelemetry Trace API reached stability in early 2021. It follows
  [Semantic Versioning 2.0](/docs/specs/otel/versioning-and-stability/).
- Use the earliest stable OpenTelemetry API (1.0.\*) and avoid updating it
  unless you have to use new features.

## 82
QUESTION: How does passing context between requests let metrics be broken down by the chain of calls?
QUOTE: In the case of metrics, context propagation enables you to aggregate
measurements in that context. For example, instead of only looking at the
response time of all the `GET /product` requests, you can also get metrics for
combinations of `POST /cart/add > GET /product` and
`GET /checkout > GET /product`.

## 83
QUESTION: When is the keep-or-drop choice made in head sampling, and what does it not look at?
QUOTE: Head sampling is a sampling technique used to make a sampling decision as early
as possible. A decision to sample or drop a span or trace is not made by
inspecting the trace as a whole.

## 84
QUESTION: How are the values filled into a prompt template recorded on a span?
QUOTE: **[9] `gen_ai.prompt.variable`:** Prompt templates are parameterized with variables that are filled in
at runtime. This attribute records the variable values passed to the
template. The attribute name defines the variable name, and the
attribute value is the variable value serialized as a string.

## 85
QUESTION: What properties should each retrieved document object carry when they are available?
QUOTE: **[8] `gen_ai.retrieval.documents`:** Each document object SHOULD contain the following properties when available:
`id` (string): A unique identifier for the document, `score` (double): The relevance score of the document

## 86
QUESTION: Can one output message mix parts from several generated choices?
QUOTE: **[36] `gen_ai.output.messages`:** Each message represents a single output choice/candidate generated by
the model. Each message corresponds to exactly one generation
(choice/candidate) and vice versa - one choice cannot be split across
multiple messages or one message cannot contain parts from multiple choices.

## 87
QUESTION: Why is the input-messages attribute absent from a span that records a fetched response?
QUOTE: **[12] `gen_ai.system_instructions`:** Only content carried on the fetched response is recorded. A fetched response contains the system instructions, output messages, and tool definitions but not the original input messages, so `gen_ai.input.messages` is not set on this span.

## 88
QUESTION: What does the workflow duration metric measure from and to?
QUOTE: **[1]:** Represents the end-to-end duration of a workflow execution, measured from the point where application code initiates the workflow to the point where the workflow completes, independent of workflow complexity.
A workflow is a coordinated process composed of multiple agents or other operations involving generative AI.

## 89
QUESTION: Why is the response identifier left out of the span name?
QUOTE: **Span name** SHOULD be `{gen_ai.operation.name}`. The response identifier is high
cardinality, so it is not included in the span name. Semantic conventions for individual
GenAI providers MAY specify a different span name format.

## 90
QUESTION: When should instrumentation set the flag that shows the conversation was compacted?
QUOTE: **[15] `gen_ai.conversation.compacted`:** This attribute is a positive indicator of context compaction. Instrumentations
SHOULD set it to `true` only when they can reliably determine that context
compaction was applied. Instrumentations SHOULD NOT set it to `false`; they
SHOULD leave it unset otherwise.

## 91
QUESTION: What may the prompt version string look like, and which system's identifier should it match?
QUOTE: **[8] `gen_ai.prompt.version`:** The version string can follow any versioning scheme chosen by the
application (e.g., SemVer, date-based, or platform-specific tags).
When a prompt management system is in use, this SHOULD match the
version identifier used by that system.

## 92
QUESTION: Can strictness and constraints be applied to the extra types that come in a separate package?
QUOTE: On top of these, Pydantic provides extra types, either [directly in the library](../api/types.md)
(e.g. [`SecretStr`][pydantic.types.SecretStr]) or in the [`pydantic-extra-types`](https://github.com/pydantic/pydantic-extra-types)
external library. These are implemented using the patterns described in the [custom types](#custom-types) section.
Strictness and constraints *can't* be applied on them.

## 93
QUESTION: Why is it better to use concrete container types than abstract ones for validation speed?
QUOTE: Besides, using these abstract types can also lead to [poor validation performance](./performance.md#sequence-vs-list-or-tuple-with-mapping-vs-dict), and in general using concrete container types
will avoid unnecessary checks.

## 94
QUESTION: How can you find which record in a large file failed validation after an unattended run?
QUOTE: With two records, spotting a bad one is easy. In a pipeline processing files with thousands of records,
the error's `loc` gives you the index of the failing one, and if the pipeline runs unattended,
[Logfire](../errors/troubleshooting.md) records failed validations with their field locations and
rejected values, so you can find the offending record after the run.

## 95
QUESTION: What are the three settings for extra data, and which one is the default?
QUOTE: * `'ignore'`: Providing extra data is ignored (the default).
* `'forbid'`: Providing extra data is not permitted.
* `'allow'`: Providing extra data is allowed and stored in the `__pydantic_extra__` dictionary attribute.
  The `__pydantic_extra__` can explicitly be annotated to provide validation for extra fields.

## 96
QUESTION: Why might serializing every value by its runtime type be unwanted?
QUOTE: However, do note that the *serialize as any* behavior will apply to *all* values, not only the values where duck typing
is relevant. You may want to prefer using the `SerializeAsAny` annotation when required instead.

## 97
QUESTION: What can a plugin around the validator observe for each validation?
QUOTE: Pydantic wraps the `SchemaValidator` in a plugin layer: when plugins are installed, each of the
`validate_python`, `validate_json`, and `validate_strings` calls can be intercepted, letting a plugin
observe the input, result, and any error for every validation.

## 98
QUESTION: How do you add type hints to a fixture function?
QUOTE: To type fixtures in pytest, just add normal types to the fixture functions -- there is nothing special that needs to be done just because of the `fixture` decorator.

## 99
QUESTION: How can warning filters be applied only to chosen tests, classes or modules?
QUOTE: You can use the @pytest.mark.filterwarnings mark to add warning filters to specific test items,
allowing you to have finer control of which warnings should be captured at test, class or
even module level:

## 100
QUESTION: Which option makes pytest use a folder you name directly as the base for temporary directories?
QUOTE: - When the `--basetemp` option is used (e.g. `pytest --basetemp=mydir`),
  it will be used directly as base temporary directory:

## 101
QUESTION: Does the sequence in which test setup helpers are written decide the sequence they run in, and what should you rely on?
QUOTE: Names of fixtures or tests, where they're defined, the order they're defined in,
and the order fixtures are requested in have no bearing on execution order
beyond coincidence. While pytest will try to make sure coincidences like these
stay consistent from run to run, it's not something that should be depended on.

## 102
QUESTION: What does the environment test automation helper do, and when would you consider it?
QUOTE: Once you are done with your work and want to make sure that your actual
package passes all tests you may want to look into tox, the
virtualenv test automation tool.
`tox` helps you to setup virtualenv environments with pre-defined
dependencies and then executing a pre-configured test command with
options.

## 103
QUESTION: How can you stop plugins from loading on their own and allow only chosen ones?
QUOTE: Disable plugin auto-loading through entry point packaging metadata.
Only plugins explicitly specified in `-p` or env var `PYTEST_PLUGINS` will be loaded.

## 104
QUESTION: A repeat count of six: how many occurrences must match, and what happens with five?
QUOTE: `{m}`
   Specifies that exactly *m* copies of the previous RE should be matched; fewer
   matches cause the entire RE not to match.  For example, `a{6}` will match
   exactly six `'a'` characters, but not five.

## 105
QUESTION: What happens during the last phase of the interpreter's life that can run user-defined destructor code?
QUOTE: interpreter shutdown
   When asked to shut down, the Python interpreter enters a special phase
   where it gradually releases all allocated resources, such as modules
   and various critical internal structures.  It also makes several calls
   to the garbage collector. This can trigger
   the execution of code in user-defined destructors or weakref callbacks.

## 106
QUESTION: How can you avoid a blocking connect on a socket, and how do you tell it has succeeded?
QUOTE: You can use the `connect_ex` method
to avoid creating an exception.
It will just return the errno value.
To poll, you can call `connect_ex` again later
-- `0` or `errno.EISCONN` indicate that you're connected -- or you can pass this
socket to `select.select` to check if it's writable.

## 107
QUESTION: Which copy function is used by default when moving a directory across file systems?
QUOTE: If *copy_function* is given, it must be a callable that takes two arguments,
*src* and the destination, and will be used to copy *src* to the destination
if `os.rename` cannot be used.  If the source is a directory,
`copytree` is called, passing it the *copy_function*. The
default *copy_function* is `copy2`.

## 108
QUESTION: What is the default spacing between tab positions when expanding tabs in a byte sequence?
QUOTE: Return a copy of the sequence where all ASCII tab characters are replaced
by one or more ASCII spaces, depending on the current column and the given
tab size.  Tab positions occur every *tabsize* bytes (default is 8,
giving tab positions at columns 0, 8, 16 and so on).  To expand the
sequence, the current column is set to zero and the sequence is examined
byte by byte.

## 109
QUESTION: What states can a future be in, and what everyday device is it compared to?
QUOTE: A future has a few important attributes. One is its state, which can be either
"pending", "cancelled", or "done".
Another is its result, which is set when the state transitions to done.
Unlike a coroutine, a future does not represent the actual computation to be
done; instead, it represents the status and result of that computation, kind of
like a status light (red, yellow, or green) or indicator.

## 110
QUESTION: What is a dialect in the CSV module, and how is one passed to a reader or writer?
QUOTE: To make it easier to specify the format of input and output records, specific
formatting parameters are grouped together into dialects.  A dialect is a
subclass of the `Dialect` class containing various attributes
describing the format of the CSV file.  When creating `reader` or
`writer` objects, the programmer can specify a string or a subclass of
the `Dialect` class as the dialect parameter.

## 111
QUESTION: What does a dictionary view do after its dictionary is modified, and how do you turn it into a full list?
QUOTE: dictionary view
   The objects returned from `dict.keys`, `dict.values`, and
   `dict.items` are called dictionary views. They provide a dynamic
   view on the dictionary’s entries, which means that when the dictionary
   changes, the view reflects these changes. To force the
   dictionary view to become a full list use `list(dictview)`.  See
   dict-views.

## 112
QUESTION: How can you reliably tell whether an object can be iterated?
QUOTE: Checking `isinstance(obj, Iterable)` detects classes that are registered
as `Iterable` or that have an `__iter__` method,
but it does
not detect classes that iterate with the `__getitem__` method.
The only reliable way to determine whether an object is iterable
is to call `iter(obj)`.

## 113
QUESTION: How do you prevent attribute names from colliding with those of subclasses?
QUOTE: To avoid name clashes with subclasses, use two leading underscores to
invoke Python's name mangling rules.

## 114
QUESTION: Why does the style guide recommend importing with full package paths?
QUOTE: - Absolute imports are recommended, as they are usually more readable
  and tend to be better behaved (or at least give better error
  messages) if the import system is incorrectly configured (such as
  when a directory inside a package ends up on `sys.path`):

## 115
QUESTION: Why does the recall measure for retrieved context always need something to compare against?
QUOTE: Since it is about not missing anything, calculating context recall always requires a reference to compare against. The LLM-based Context Recall metric uses `reference` as a proxy to `reference_contexts`, which makes it easier to use as annotating reference contexts can be very time-consuming.

## 116
QUESTION: Over what range does the overlap-based text metric score, and does it use a language model?
QUOTE: The `RougeScore` score is a set of metrics used to evaluate the quality of natural language generations. It measures the overlap between the generated `response` and the `reference` text based on n-gram recall, precision, and F1 score. ROUGE score ranges from 0 to 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

## 117
QUESTION: What does the identifier-based recall measure compare, and what types of identifiers does it accept?
QUOTE: The metric computes recall using retrieved_context_ids and reference_context_ids, with values ranging between 0 and 1. Higher values indicate better performance. It works with both string and integer IDs.

## 118
QUESTION: In which release is the older metrics interface planned for removal, and when does its deprecation begin?
QUOTE: !!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

## 119
QUESTION: Why is a model scoring its own factual output unreliable, and what replaces that step in CRITIC?
QUOTE: Self-Refine's weakness: the feedback step is an LLM scoring itself. For factual claims this is unreliable (a hallucination often looks convincing to the model that produced it). CRITIC replaces `feedback(task, output)` with `verify(task, output, tools)` where `tools` includes:

## 120
QUESTION: How many cases are in the production sample tier of the test suite, and how often should it be refreshed?
QUOTE: **Tier 3: Distribution Sample (100-200 cases)**
- Random sample from production traffic (anonymized)
- Refresh monthly to track distribution shift
- Weight by frequency -- common queries matter more

## 121
QUESTION: Which field of a tool definition matters most to the model, and why?
QUOTE: The description is the most important field. The model reads it to decide when to use the tool.

## 122
QUESTION: How do the four observability tools differ in the question each is best at answering?
QUOTE: They don't solve the same problem. LangSmith answers "why did this LangGraph run fail?" Phoenix answers "is my RAG pipeline drifting?" Helicone answers "which app is burning tokens?" Langfuse answers "can I self-host the whole thing?" Different tools, different audiences.

## 123
QUESTION: In which orchestration pattern do specialists never communicate with one another, and who dispatches the work?
QUOTE: | Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatches to specialists; they don't talk to each other |
| Swarm | "Peer-to-peer" | Direct handoffs via shared tools; no central router |
| Hierarchical | "Supervisors of supervisors" | Nested subgraphs for large populations |

## 124
QUESTION: Which real systems are given as examples of prompt-extraction and injection attacks?
QUOTE: These are not theoretical. Bing Chat's system prompt was extracted on day one of public preview. ChatGPT plugins were exploited to exfiltrate conversation data. Google Bard was tricked into endorsing phishing sites through indirect injection in Google Docs.

## 125
QUESTION: After how many days open does an issue become eligible for a contributor reward?
QUOTE: 1. Maintainer time on new lessons and on the issue queue.
2. Site hosting, domain, and CDN (Vercel + custom domains).
3. Diagram authoring tools, font licensing, design assets.
4. One-time research or content fees for guest lesson reviewers when a phase covers
   territory outside the maintainer's depth.
5. Contributor bounties on specific issues that have been open longer than 30 days.

## 126
QUESTION: If a company has fifty internal functions, how many should a tool-calling model be allowed to reach?
QUOTE: **Rule 2: Allowlist functions.** The model can only call functions you explicitly define. Never build a generic "execute any function by name" tool. If you have 50 internal functions, expose only the 5 the user needs.

## 127
QUESTION: How deeply may objects be nested in a schema before extraction errors rise, and how should ambiguous fields be declared?
QUOTE: 1. Keep schemas flat when possible. Nested objects beyond 2 levels increase extraction errors.
2. Use enums for categorical fields. Do not rely on the model inventing the right string.
3. Make ambiguous fields required with explicit null support rather than optional. Forces the model to make a decision.
4. Add descriptions to schema properties. The model reads these as instructions.
5.

## 128
QUESTION: Which of the three editor-setting preferences ignores configuration files completely?
QUOTE: - `"editorFirst"`: Editor settings take priority over configuration files present in the workspace.
- `"filesystemFirst"`: Configuration files present in the workspace take priority over editor
    settings.
- `"editorOnly"`: Ignore configuration files entirely i.e., only use editor settings.

## 129
QUESTION: Which component walks the syntax tree when a new lint rule is added, and what must you change so the violation fires?
QUOTE: To trigger the violation, you'll likely want to augment the logic in `crates/ruff_linter/src/checkers/ast.rs`
to call your new function at the appropriate time and with the appropriate inputs. The `Checker`
defined therein is a Python AST visitor, which iterates over the AST, building up a semantic model,
and calling out to lint rule analyzer functions as it goes.

## 130
QUESTION: What happens to the line length set in project settings files when a command-line override is given?
QUOTE: Specifying `--config "line-length=90"` will override the `line-length`
setting from *all* configuration files detected by Ruff,
including configuration files discovered in subdirectories.

## 131
QUESTION: Does the experimental-features switch guarantee that the gated features will stay?
QUOTE: The preview mode is _not_ intended to gate access to work that is incomplete or features that we are _likely to remove._ However, **we reserve the right to make changes to _any_ behavior gated by the mode** including the removal of preview features or rules.

## 132
QUESTION: Is an editable install still allowed when binary-only installs are demanded?
QUOTE: Both pip and uv allow editable requirements to be built and installed even when `--only-binary` is
provided. For example, `uv pip install -e . --only-binary :all:` is allowed.

## 133
QUESTION: From which version can users choose pip-style behaviour for packages found on several indexes, and through which two means?
QUOTE: As of v0.1.39, users can opt in to `pip`-style behavior for multiple indexes via the
`--index-strategy` command-line option, or the `UV_INDEX_STRATEGY` environment variable, which
supports the following values:

## 134
QUESTION: How can you add a package from inside a notebook so that the project files are not changed?
QUOTE: When creating a notebook, select the `project` kernel from the dropdown. Then use `!uv add pydantic`
to add `pydantic` to the project's dependencies, or `!uv pip install pydantic` to install `pydantic`
into the project's virtual environment without persisting the change to the project `pyproject.toml`
or `uv.lock` files. Either command will make `import pydantic` work within the notebook.

## 135
QUESTION: How do you require a backport package only for older Python versions?
QUOTE: Some dependencies are only required in specific environments, e.g., a specific Python version or
operating system. For example to install the `importlib-metadata` backport for the
`importlib.metadata` module, use `importlib-metadata >=7.1.0,<8; python_version < '3.10'`. To
install `colorama` on Windows (but omit it on other platforms), use
`colorama >=0.4.6,<5; platform_system == "Windows"`.

## 136
QUESTION: When does uv first create the virtual environment and lock file for a project?
QUOTE: A project consists of a few important parts that work together and allow uv to manage your project.
In addition to the files created by `uv init`, uv will create a virtual environment and `uv.lock`
file in the root of your project the first time you run a project command, i.e., `uv run`,
`uv sync`, or `uv lock`.

## 137
QUESTION: Which index does uv use by default, and how can others be configured?
QUOTE: By default, uv uses the [Python Package Index (PyPI)](https://pypi.org) for dependency resolution
and package installation. However, uv can be configured to use other package indexes, including
private indexes, via the `[[tool.uv.index]]` configuration option (and `--index`, the analogous
command-line option).

## 138
QUESTION: Why can a cached prefix for hybrid state-space models only resume at block boundaries, and what extra checkpoint helps?
QUOTE: With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

## 139
QUESTION: What are the default values of the two settings for graph capture of the multimodal encoder?
QUOTE: * `cudagraph_mm_encoder` (`bool`, default `False`) — enable CUDA Graph capture for multimodal encoder. When enabled, captures the full encoder forward as a CUDA Graph for each token budget level.
* `encoder_cudagraph_token_budgets` (`list[int]`, default `[]`) — token budget levels for capture. If empty (default), auto-inferred from model architecture as power-of-2 levels.

## 140
QUESTION: Which environment variable helps find the GPU kernel causing a failure?
QUOTE: - `export VLLM_LOGGING_LEVEL=DEBUG` to turn on more logging.
- `export VLLM_LOG_STATS_INTERVAL=1.` to get log statistics more frequently for tracking running queue, waiting queue and cache hit states.
- `export CUDA_LAUNCH_BLOCKING=1` to identify which CUDA kernel is causing the problem.
- `export NCCL_DEBUG=TRACE` to turn on more logging for NCCL.

## 141
QUESTION: Into which language can the speech models translate, and from how many languages?
QUOTE: Our Translation API is compatible with [OpenAI's Translations API](https://platform.openai.com/docs/api-reference/audio/createTranslation);
you can use the [official OpenAI Python client](https://github.com/openai/openai-python) to interact with it.
Whisper models can translate audio from one of the 55 non-English supported languages into English.

## 142
QUESTION: What can be given beside the count when limiting multimodal inputs per prompt?
QUOTE: `limit_mm_per_prompt` also accepts configurable options per modality. In the configurable form, you still specify `count`, and you may optionally provide size hints that control how vLLM profiles and reserves memory for your multi‑modal inputs. This helps you tune memory for the actual media you expect, instead of the model’s absolute maxima.

## 143
QUESTION: How can you check that AMD optimized kernels are being used without setting any flag?
QUOTE: vLLM auto-detects the platform and routes linear layers through ZenDNN-optimized kernels - no flag needed. To verify it is engaged, look for the platform-selection line in the server's startup logs:

## 144
QUESTION: When does the sampler skip some post-processing steps to save work, and why is that safe?
QUOTE: At sampling time, the sampler checks whether all requests in the persistent batch employ greedy sampling. If that is the case, the sampler saves compute by skipping "argmax-invariant" logits processors. Here, "argmax" is shorthand for the token ID with the highest logit value in a given row of the logits tensor (i.e. the token which the model weighted the highest for a given request).

## 145
QUESTION: When can a custom token-adjusting plugin be loaded without being passed explicitly?
QUOTE: Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to the `LLM` or `AsyncLLM` constructors or to the vLLM server explicitly at initialization time if your logits processor is exposed as an entry point.

## 146
QUESTION: What does the option to offload graph memory during sleep depend on, and what is its default?
QUOTE: By default, CUDA graph memory stays on the GPU while asleep. With
`sleep_mode_offload_cudagraph=True` (off by default), CUDA graphs are captured
into a cuMem pool that sleep backs up to CPU memory at both levels and any wake
restores in place, so graphs are replayed, not recaptured. It needs the default
`cumem` backend, CUDA or ROCm, and CUDA graphs; otherwise it has no effect.
