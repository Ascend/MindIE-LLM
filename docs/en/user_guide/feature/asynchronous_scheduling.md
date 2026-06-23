# Asynchronous Scheduling

Asynchronous scheduling is a scheduling algorithm that overlaps data processing latency with model inference time. It addresses the inefficiency of serial CPU and NPU usage in synchronous inference scenarios.
In synchronous inference scenarios, the inference process can be divided into the following three phases based on execution on the CPU or NPU:

- Request scheduling and preparation phase (executed on the CPU)
- Model inference and sampling phase (executed on the NPU)
- Result judgment and response phase (executed on the CPU)

CPU and NPU tasks can be executed concurrently because they use different computing resources. To improve the overall resource utilization and throughput of the system, MindIE leverages the preceding characteristics and uses multiple threads to implement asynchronous scheduling.
It is worth noting that in this mode, requests that have entered the end of sequence (EOS) state will be calculated again, which may slightly waste NPU computing and memory resources.
Therefore, this feature is typically applicable to scenarios where `max_batch_size` is large and the output length is long.

## Constraints

- This feature is supported in the prefill-decode co-location and prefill-decode disaggregation scenarios.
- This feature cannot be used with Look Ahead or Memory Decoding.
- Currently, this feature does not support postprocessing parameters related to multi-sequence inference, such as `n`, `best_of`, and `use_beam_search`.

## Inference

1. Set the following environment variable to enable asynchronous scheduling.

    ```bash
    export MINDIE_ASYNC_SCHEDULING_ENABLE=1
    ```

    > [!NOTE]NOTE
    > In the prefill-decode disaggregation scenario, perform this operation only on the decode node.

2. Open the `config.json` file of the server.

    ```bash
    cd {MindIE_installation_directory}/mindie_llm/
    vi conf/config.json
    ```

3. Set serving parameters. For details about the serving parameters, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md).
4. Start the service.

    ```bash
    mindie_llm_server
    ```

5. Use the AISBench tool to start tuning. For details about the AISBench tool, see "Auxiliary Tools" > "Performance/Accuracy Test Tool" in *MindIE Motor Development Guide*.
