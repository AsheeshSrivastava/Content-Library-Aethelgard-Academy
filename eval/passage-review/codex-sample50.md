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

## 7
QUESTION: Which model judged faithfulness throughout the study, and with what sampling settings?
QUOTE: All faithfulness evaluations (both the factorial ANOVA and the profile comparisons) use the same GPT-5.1 judge model with identical RAGAS v0.4 configuration (T{=}0, deterministic seed, batch_size{}=4).

## 8
QUESTION: In the executed subset, how many of the main effects and two-way interactions can be estimated, and what is the range of cell sizes?
QUOTE: The 167 executed configurations cover all factor levels in 7 of 8 dimensions and 3 of 4 levels in the enrichment dimension (D3). All 22 main effects and all 28 two-way interactions are estimable from this subset. However, the design is unbalanced (cell sizes range from 1 to 7), which affects the precision of individual interaction estimates.

## 10
QUESTION: What does Toggle do during budget-limited training phases in Kimi K2.5?
QUOTE: Kimi K2.5 introduces Toggle, a training heuristic that alternates between inference-time scaling and budget-constrained optimization. During budget-limited phases, the model is trained to solve problems within a task-dependent token budget, producing more concise chain-of-thought reasoning.

## 12
QUESTION: What lesson do the authors draw about relying on compact models to produce reasoning explanations?
QUOTE: This approach shows a clever use of small language models for rationale generation. The results are remarkable given that larger language models are preferred for this capability over smaller ones. Decomposing tasks in this way is something developers should think deeply about. Not everything needs to be done by the large models.

## 19
QUESTION: What do the optimizers actually change in a program to improve a metric?
QUOTE: **What do DSPy optimizers tune?** Or, _what does compiling actually do?_ Each optimizer is different, but they all seek to maximize a metric on your program by updating prompts or LM weights.

## 20
QUESTION: Which adapter suits a model that formats its answers unreliably?
QUOTE: Same Signature, different prompt shape, no Signature changes. LMs vary a lot in what they prefer to read and produce: a model trained on instruction-following with markers loves ChatAdapter’s `[[ ## field ## ]]` format; a model with native structured-output mode does best with JSONAdapter; a reasoning model that formats unreliably wants TwoStepAdapter.

## 22
QUESTION: What risk is shown when a model that includes a password is used for both input and output?
QUOTE: Now, whenever a browser is creating a user with a password, the API will return the same password in the response.

## 24
QUESTION: In the tutorial on returning several model types, why does putting two model classes joined by a vertical bar directly into the response model argument raise an error?
QUOTE: But if we put that in the assignment `response_model=PlaneItem | CarItem` we would get an error, because Python would try to perform an **invalid operation** between `PlaneItem` and `CarItem` instead of interpreting that as a type annotation.

## 31
QUESTION: Which model does the NVIDIA ranker use if none is given?
QUOTE: `NvidiaRanker` ranks `Documents` based on semantic relevance to a specified query. It uses ranking models provided by [NVIDIA NIMs](https://ai.nvidia.com). If you don't set the `model` parameter, the hosted default `nv-rerank-qa-mistral-4b:1` is used.

## 32
QUESTION: Which optional search settings can be passed to the MongoDB Atlas full-text retriever when it runs?
QUOTE: When running the component, you can specify more optional parameters such as `fuzzy` or `synonyms`, `match_criteria`, `score`. Check out our [MongoDB Atlas](/reference/integrations-mongodb-atlas) API Reference for more details on all parameters.

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

## 42
QUESTION: Which cache class do models use by default, and how does its size behave as text is produced?
QUOTE: The [`DynamicCache`] is the default cache class for all models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.

## 44
QUESTION: How many weights does AQLM treat as one group, and what do they get expressed as?
QUOTE: Additive Quantization of Language Models ([AQLM](https://huggingface.co/papers/2401.06118)) quantizes multiple weights together and takes advantage of interdependencies between them. AQLM represents groups of 8-16 weights as a sum of multiple vector codes.

## 45
QUESTION: Which image processor backend is the default, and why is it preferred?
QUOTE: - [`BaseImageProcessor`] — abstract base class (for backward compatibility only; do not instantiate directly)
  - [`TorchvisionBackend`] — the **default** [torchvision-backed](https://pytorch.org/vision/stable/index.html) backend. GPU-accelerated and significantly faster than the PIL backend. All models expose a `<Model>ImageProcessor` class that inherits from it.

## 49
QUESTION: Why are models with custom code skipped when looking up a conversion mapping?
QUOTE: Custom code models are skipped before any lookup happens. A sub-model counts as custom code when [`~PreTrainedModel.is_custom_code`] returns `True`, which covers models loaded from the Hub with `trust_remote_code=True` and any model class not in Transformers. Custom code models don't inherit a conversion mapping because its class name or `model_type` might collide with a native model.

## 51
QUESTION: In what order does FastAPI run the cleanup code of dependencies that use yield when they are nested?
QUOTE: **FastAPI** will make sure that the "exit code" in each dependency with `yield` is run in the correct order.

## 54
QUESTION: Which callback counts tokens for cost analysis?
QUOTE: Tokens are counted using the `TokenCountingHandler` callback. See the [example notebook](/python/examples/observability/tokencountinghandler) for details on the setup.

## 57
QUESTION: What happens to a document's attributes when a node parser breaks it into chunks?
QUOTE: Node parsers are a simple abstraction that take a list of documents, and chunk them into `Node` objects, such that each node is a specific chunk of the parent document. When a document is broken into nodes, all of it's attributes are inherited to the children nodes (i.e. `metadata`, text and metadata templates, etc.).

## 59
QUESTION: What minimum version of the Python SDK is needed for the client tutorial?
QUOTE: - Mac or Windows computer
- Latest Python version installed
- Latest version of `uv` installed
- You must use the Python MCP SDK 2.0.0 or higher

## 62
QUESTION: Which two fields handle version negotiation in the discovery exchange?
QUOTE: 1. **Protocol Version Selection**: The `io.modelcontextprotocol/protocolVersion` field declares the version the client is speaking on this request, and `supportedVersions` in the response lists the versions the server accepts.

## 64
QUESTION: What do roots tell servers, and how does a server learn about changes to them?
QUOTE: Roots are exclusively filesystem paths and always use the `file://` URI scheme. They help servers understand project boundaries, workspace organization, and accessible directories. The roots list can change as users work with different projects or folders. Servers pick up the updated boundaries the next time they request the roots list.

## 68
QUESTION: Why does tampering with a signed payload make verification fail?
QUOTE: Verification fails. The signature was computed over the canonical bytes of the original payload; modifying any field changes those bytes, which makes the signature invalid. The attacker would need the private key to produce a fresh valid signature, which they do not have.

## 70
QUESTION: How should an agent's access to important systems be limited?
QUOTE: **Mitigation**: AI agents should have access to systems on a need-only basis to prevent these types of attacks. Communication between the agent and system should also be secure. Implementing authentication and access control is another way to protect this information.

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

## 78
QUESTION: Does the SharePoint URL inspection command change the ChatGPT allowlist?
QUOTE: This command calls Microsoft Graph and prints each resolved site identifier and
the deduplicated collection GUIDs. It does not read or modify the ChatGPT
workspace allowlist. Tenant-root URLs and percent-encoded paths, such as
`https://contoso.sharepoint.com/sites/Finance%20Team`, are both supported.

## 79
QUESTION: What does the gating network in a mixture of experts do, and what ensures the weights sum to one?
QUOTE: - **Gating Network:** The gating network is responsible for dynamically selecting which experts should be consulted for a given input. It assigns weights to each expert's output, determining their contribution to the final prediction. The gating network typically uses a softmax function to produce these weights, ensuring they sum to one.

## 82
QUESTION: How does passing context between requests let metrics be broken down by the chain of calls?
QUOTE: In the case of metrics, context propagation enables you to aggregate
measurements in that context. For example, instead of only looking at the
response time of all the `GET /product` requests, you can also get metrics for
combinations of `POST /cart/add > GET /product` and
`GET /checkout > GET /product`.

## 84
QUESTION: How are the values filled into a prompt template recorded on a span?
QUOTE: **[9] `gen_ai.prompt.variable`:** Prompt templates are parameterized with variables that are filled in
at runtime. This attribute records the variable values passed to the
template. The attribute name defines the variable name, and the
attribute value is the variable value serialized as a string.

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

## 95
QUESTION: What are the three settings for extra data, and which one is the default?
QUOTE: * `'ignore'`: Providing extra data is ignored (the default).
* `'forbid'`: Providing extra data is not permitted.
* `'allow'`: Providing extra data is allowed and stored in the `__pydantic_extra__` dictionary attribute.
  The `__pydantic_extra__` can explicitly be annotated to provide validation for extra fields.

## 100
QUESTION: Which option makes pytest use a folder you name directly as the base for temporary directories?
QUOTE: - When the `--basetemp` option is used (e.g. `pytest --basetemp=mydir`),
  it will be used directly as base temporary directory:

## 106
QUESTION: How can you avoid a blocking connect on a socket, and how do you tell it has succeeded?
QUOTE: You can use the `connect_ex` method
to avoid creating an exception.
It will just return the errno value.
To poll, you can call `connect_ex` again later
-- `0` or `errno.EISCONN` indicate that you're connected -- or you can pass this
socket to `select.select` to check if it's writable.

## 110
QUESTION: What is a dialect in the CSV module, and how is one passed to a reader or writer?
QUOTE: To make it easier to specify the format of input and output records, specific
formatting parameters are grouped together into dialects.  A dialect is a
subclass of the `Dialect` class containing various attributes
describing the format of the CSV file.  When creating `reader` or
`writer` objects, the programmer can specify a string or a subclass of
the `Dialect` class as the dialect parameter.

## 118
QUESTION: In which release is the older metrics interface planned for removal, and when does its deprecation begin?
QUOTE: !!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

## 119
QUESTION: Why is a model scoring its own factual output unreliable, and what replaces that step in CRITIC?
QUOTE: Self-Refine's weakness: the feedback step is an LLM scoring itself. For factual claims this is unreliable (a hallucination often looks convincing to the model that produced it). CRITIC replaces `feedback(task, output)` with `verify(task, output, tools)` where `tools` includes:

## 131
QUESTION: Does the experimental-features switch guarantee that the gated features will stay?
QUOTE: The preview mode is _not_ intended to gate access to work that is incomplete or features that we are _likely to remove._ However, **we reserve the right to make changes to _any_ behavior gated by the mode** including the removal of preview features or rules.

## 133
QUESTION: From which version can users choose pip-style behaviour for packages found on several indexes, and through which two means?
QUOTE: As of v0.1.39, users can opt in to `pip`-style behavior for multiple indexes via the
`--index-strategy` command-line option, or the `UV_INDEX_STRATEGY` environment variable, which
supports the following values:

## 136
QUESTION: When does uv first create the virtual environment and lock file for a project?
QUOTE: A project consists of a few important parts that work together and allow uv to manage your project.
In addition to the files created by `uv init`, uv will create a virtual environment and `uv.lock`
file in the root of your project the first time you run a project command, i.e., `uv run`,
`uv sync`, or `uv lock`.

## 138
QUESTION: Why can a cached prefix for hybrid state-space models only resume at block boundaries, and what extra checkpoint helps?
QUOTE: With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

## 140
QUESTION: Which environment variable helps find the GPU kernel causing a failure?
QUOTE: - `export VLLM_LOGGING_LEVEL=DEBUG` to turn on more logging.
- `export VLLM_LOG_STATS_INTERVAL=1.` to get log statistics more frequently for tracking running queue, waiting queue and cache hit states.
- `export CUDA_LAUNCH_BLOCKING=1` to identify which CUDA kernel is causing the problem.
- `export NCCL_DEBUG=TRACE` to turn on more logging for NCCL.

## 142
QUESTION: What can be given beside the count when limiting multimodal inputs per prompt?
QUOTE: `limit_mm_per_prompt` also accepts configurable options per modality. In the configurable form, you still specify `count`, and you may optionally provide size hints that control how vLLM profiles and reserves memory for your multi‑modal inputs. This helps you tune memory for the actual media you expect, instead of the model’s absolute maxima.

## 144
QUESTION: When does the sampler skip some post-processing steps to save work, and why is that safe?
QUOTE: At sampling time, the sampler checks whether all requests in the persistent batch employ greedy sampling. If that is the case, the sampler saves compute by skipping "argmax-invariant" logits processors. Here, "argmax" is shorthand for the token ID with the highest logit value in a given row of the logits tensor (i.e. the token which the model weighted the highest for a given request).

## 145
QUESTION: When can a custom token-adjusting plugin be loaded without being passed explicitly?
QUOTE: Once your package is installed, your custom logits processor will be loaded automatically whenever vLLM is initialized. You do *not* need to pass the custom logits processor to the `LLM` or `AsyncLLM` constructors or to the vLLM server explicitly at initialization time if your logits processor is exposed as an entry point.
