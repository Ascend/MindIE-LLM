# Environment Variable Description

After MindIE LLM is installed, the process-level environment variable setting script `set_env.sh` is provided to automatically set environment variables.

## Environment Variables of the `set_env.sh` Script

**Table 1** Environment variables of the `set_env.sh` script

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|**MindIE LLM environment variables**|-|-|-|
|MINDIE_LLM_HOME_PATH|MindIE LLM home path.|N/A|N/A|
|MINDIE_LLM_RECOMPUTE_THRESHOLD|Recomputation threshold in MindIE LLM.|[0,1]|0.5|
|PYTORCH_INSTALL_PATH|Installation path of the third-party component torch. You can run the following command to obtain the path: `python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.file)))'`|N/A|N/A|
|PYTORCH_NPU_INSTALL_PATH|Installation path of the third-party component torch_npu. To obtain the value, run `python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))'`.|N/A|N/A|
|**ATB Models environment variables**|-|-|-|
|ATB_OPERATION_EXECUTE_ASYNC|Asynchronous scheduling of ATB graphs. By default, level-2 pipeline is used. When the number of CPUs is not limited, you can enable level-3 pipeline for performance tuning.|`0`: disabled<br>`1`: level-2 pipeline<br>`2`: level-3 pipeline|1|
|ATB_SPEED_HOME_PATH|(Required) Environment variable of the lib path of ATB Models.|The value must be the lib path of ATB Models.|None|
|HCCL_INTRA_PCIE_ENABLE|Whether to enable All2All layered communication and INT8 communication features. This function can be enabled only when both `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` are enabled. For more information about these two environment variables, see "Collective Communication" in *CANN Environment Variable Reference*. You are advised to enable this function on the Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server in the "Combine INT8" operator scenario of the MoE model to improve performance.|`0`: disabled<br>`1`: enabled|N/A|
|HCCL_INTRA_ROCE_ENABLE|Whether to enable All2All layered communication and INT8 communication features. This function can be enabled only when both `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` are enabled. For more information about these two environment variables, see "Collective Communication" in *CANN Environment Variable Reference*. You are advised to enable this function on the Atlas 800I A2 inference server and Atlas 800I A3 SuperPoD server in the "Combine INT8" operator scenario of the MoE model to improve performance.|`0`: disabled; <br>`1`: enable.|N/A|
|**Ascend Extension for PyTorch environment variables**|-|-|-|
|MASTER_IP|Host IP address for multi-server serving.|If the value is not empty, the IP address must be valid.|None|
|MASTER_PORT|Host port for multi-server serving.|If the value is not empty, the port number ranges from 0 to 65535.|None|

## Other Environment Variables

For details about server-related environment variables, see **Table 2**.

**Table 2** Server-related environment variables

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|MINDIE_LLM_HOME_PATH|Server installation path.|Path parameter|/usr/local/Ascend/mindie/latest/mindie-service|
|MIES_CONFIG_JSON_PATH|Path to the `config.json` file. If the environment variable exists, its value is read. If the environment variable does not exist, the value from the `${MINDIE_LLM_HOME_PATH}/conf/config.json` file is read.|Path parameter|N/A|
|MIES_CONTAINER_IP|Container IP address, which is configured during container deployment. IP address bound to the service-plane RESTful API provided by Endpoint and IP address used for gRPC communication in multi-server inference scenarios. This environment variable needs to be set for multi-server inference.|IPv4 address|N/A|
|MIES_CONTAINER_MANAGEMENT_IP|IP address bound to the internal RESTful API and provided by EndPoint.|IPv4 address|N/A|
|MIES_MEMORY_DETECTOR_MODE|Whether to detect memory status by dotting.|`0`: disabled<br>`1`: enabled|0|
|MIES_PROFILER_MODE|Whether to detect performance status by dotting.|`0`: disabled<br>`1`: enabled|0|
|LD_LIBRARY_PATH|Path of lib.|Path parameter|${MINDIE_LLM_HOME_PATH}/lib:${LD_LIBRARY_PATH}|
|ASCEND_SLOG_PRINT_TO_STDOUT|CANNDEV log printing switch|`1`: Print logs.<br>`0`: Write logs to the `~/ascend` directory.|0|
|ASCEND_GLOBAL_LOG_LEVEL|CANNDEV log level.|**0**: debug<br>**1**: info<br>**2**: warn<br>**3**: error|3|
|ASCEND_GLOBAL_EVENT_ENABLE|Whether to enable event logging for applications.|**0**: Event log disabled log. <br>**1**: enabled|0|
|HCCL_BUFFSIZE|Size of the buffer that controls shared data between two NPUs.|≥ 1, in MB|120|
|EP_OPENSSL_PATH|After HTTPS authentication is enabled for EndPoint, this environment variable is used to specify the runtime .so file loaded by OpenSSL. This environment variable is automatically set when the EndPoint module is started. You do not need to manually set it.|Path parameter|${MINDIE_LLM_HOME_PATH}/lib|
|HSECEASY_PATH|After HTTPS authentication is enabled for EndPoint, use the HSECEASY tool to encrypt keys and passwords. This environment variable specifies the path of the runtime .so file loaded by HSECEASY.|Path parameter|${MINDIE_LLM_HOME_PATH}/lib|
|MIES_CERTS_LOG_TO_FILE|Environment variable of the certificate management tool, indicating whether logs are exported to a file.|`0`: Output logs to a file.<br>`1`: Do not output logs to a file.|0|
|MIES_CERTS_LOG_TO_STDOUT|Environment variable of the certificate management tool, indicating whether to print logs.|`0`: Do not print logs.<br>`1`: Print logs.|1|
|MIES_CERTS_LOG_LEVEL|Environment variable of the certificate management tool, which specifies the log level.|DEBUG<br>INFO<br>WARNING<br>ERROR<br>FATAL|INFO|
|MIES_CERTS_LOG_PATH|Environment variable of the certificate management tool, which specifies the log path.|Path parameter|/workspace/log/certs.log|
|DYNAMIC_AVERAGE_WINDOW_SIZE|Size of the dynamic window for dynamically collecting statistics on the average metric value in the `/metrics-json` interface.|Positive number|1000|
|MIES_SERVICE_MONITOR_MODE|Whether to enable the online metric management and control for inference serving. The `/metrics` interface can be requested only when this function is enabled.|`0`: disabled<br>`1`: enabled|0|
|LOCAL_CACHE_DIR|Temporary storage path for images received in multimodal requests.|Path parameter|~/mindie/cache|
|TOKENIZER_ENCODE_TIMEOUT|Timeout interval for truncating TOKENIZER Encode, in seconds.|[5, 300]|60|
|MINDIE_ASYNC_SCHEDULING_ENABLE|Whether to enable asynchronous scheduling.|1: enabled; other values: disabled|NA|

For details about MindIE LLM environment variables, see **Table 3**.

**Table 3** Environment variables related to MindIE\_LLM

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|HOST_IP|Host IP address.<br>IP address of the physical machine that provides the inference API. This parameter needs to be configured only for Coordinator.|N/A|N/A|
|LOCAL_RANK|Local ID of a device.|[0, ${WORLD_SIZE} - 1]|0|
|MIES_USE_MB_SWAPPER|High-performance swap switch.|`0`: disabled<br>`1`: enabled|0|
|MINDIE_CHECK_INPUTFILES_PERMISSION|Whether to verify the permission of external files, including the write permission of the file owner and others.|`0`: The permission of external files is not verified.<br>Other values or `None`: The permission of external files is verified.|None|
|MINDIE_LLM_BENCHMARK_ENABLE|Whether to enable the benchmark function of the MindIE LLM module. After the function is enabled, performance data is exported to a specified file path.|`0`: disabled<br>`1`: enabled|0|
|MINDIE_LLM_BENCHMARK_FILEPATH|Path of the performance data file generated by the benchmark function of the MindIE LLM module.|N/A|"${MINDIE_LLM_HOME_PATH}/logs/benchmark.jsonl"|
|MINDIE_LLM_BENCHMARK_RESERVING_RATIO|When the size of a performance data file exceeds the upper limit, the new data will overwrite the old data. This environment variable specifies the reserving ratio of old data. The default value is `0.1`.|[0.0, 1.0]|0.1|
|NPU_DEVICE_IDS|ID of the NPU used.|[0, *NPU ID*]<br>Example: [0, 1, 2, ...]|N/A|
|NPU_MEMORY_FRACTION|NPU memory usage, which indicates the ratio of the total graphics memory allocated to the model weights, KV cache, and workspace. The space allocated by HCCL and PyTorch is not included. You are advised to set this parameter to the minimum value that can start the service. The method is as follows: Start the service based on the default configuration. If the service cannot be started, increase the parameter value until the service can just be started. If the service is started successfully, decrease the parameter value until the service can just be started. In a word, a smaller value ensures higher service system stability on the premise that the service can be started properly.|(0.0, 1.0]. For the Kimi K2 model, the recommended value is `0.9`.|The default value is `1.0` for ATB Models and `0.8` for MindIE LLM.|
|PERFORMANCE_PREFIX_TREE_ENABLE|Whether to enable the high performance trie-tree of memory_decoding.|`0`: disabled<br>`1`: enabled|0|
|POST_PROCESSING_SPEED_MODE_TYPE|Postprocessing acceleration mode.|`0`: Disable acceleration.<br>`1`: Enable top_p approximate calculation.<br>`2`: Enable index acceleration.<br>`3`: Enable top_p approximate calculation and index acceleration at the same time.|0|
|RANK|Global ID of a device.|[0, ${WORLD_SIZE})|0|
|SOURCE_DATE_EPOCH|It eliminates the bep differences of the .whl package.|N/A|N/A|
|WORLD_SIZE|Number of devices used for inference.|[1,1048576]|N/A|

For details about the environment variables related to ATB\_Models, see **Table 4**.

**Table 4** Environment variables related to ATB\_Models

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|ATB_LLM_BENCHMARK_ENABLE|Whether to enable the function of obtaining performance data.|`0`: disabled<br>Other values: enabled|0|
|ATB_LLM_BENCHMARK_FILEPATH|Path for storing performance data.|All values|None|
|ATB_LLM_ENABLE_AUTO_TRANSPOSE|Whether to enable automatic transpose optimization of the weight right matrix.|`None` or `1`: enabled<br>Other values: disabled|None|
|ATB_LLM_HCCL_ENABLE|Whether to enable Huawei collective communication library.|N/A|N/A|
|ATB_LLM_LCOC_ENABLE|Whether to enable the communication and computation overlapping.|`None` or `1`: enabled<br>Other values: disabled|None|
|ATB_LLM_LOGITS_SAVE_ENABLE|Whether to save logits information.|`0`: no<br>Other values: yes|0|
|ATB_LLM_LOGITS_SAVE_FOLDER|Folder for saving logits information.|All values|None|
|ATB_LLM_RAZOR_ATTENTION_ENABLE|RA compression needs to be enabled.|`0`: disabled<br>`1`: enabled|0|
|ATB_LLM_RAZOR_ATTENTION_ROPE|Whether to enable the Razor attention compression algorithm of RoPE.|`0`: disabled<br>`1`: enabled|0|
|ATB_LLM_TOKEN_IDS_SAVE_ENABLE|Whether to save token information.|`0`: no<br>Other values: yes|0|
|ATB_LLM_TOKEN_IDS_SAVE_FOLDER|Folder for saving token information.|All values|None|
|ATB_PROFILING_ENABLE|Whether to collect performance profile data.|`1`: yes<br>Other values or `None`: no|None|
|ATB_USE_TILING_COPY_STREAM|Whether to enable dual-stream.|`1`: enabled<br>Other values or `None`: disabled|None|
|BIND_CPU|Whether to bind processes running on NPUs to cores based on the CPU affinity.|`None` or `1`: enabled<br>Other values: disabled|None|
|CPU_BINDING_NUM|Number of cores bound to each device.|[0, Number of CPU cores/Number of devices on NUMA]|None|
|HCCL_DETERMINISTIC|Deterministic computation of HCCL communication. You are advised to enable this function for multi-server inference.|`false`: disabled<br>`true`: enabled|Generally, the value is `true` and depends on the model.|
|IS_ALIBI_MASK_FREE|Whether to support Speculate.|`1`: enabled<br>Other values or `None`: disabled|None|
|LCCL_DETERMINISTIC|Deterministic computation of LCCL communication.|`0`: disabled<br>`1`: enabled|Generally, the value is `1` and depends on the model.|
|LONG_SEQ_ENABLE|Whether to enable the long sequence feature.|`1`: yes<br>Other values or `None`: no|None|
|MINDIE_ACLNN_CACHE_GLOBAL_COUNT|Number of global caches of `aclExecutor` and the corresponding `aclTensor` in Plugin Op.|[0, 100)|16|
|PROFILING_FILEPATH|Path of the profiling files. By default, the profiling files are saved in the profiling folder in the current path.|N/A|N/A|
|PROFILING_LEVEL|ProfilerLevel.|Level0<br>Level1<br>Level2<br>Level_none|Level0|
|RESERVED_MEMORY_GB|Size of the graphics memory pool that is dynamically allocated during model running.|[0, 64)|3|
|MINDIE_ENABLE_EXPERT_HOTPOT_GATHER|Whether to collect expert hotspot information for load balancing.|`1`: enabled<br>Other values or `None`: disabled|None|
|MINDIE_EXPERT_HOTPOT_DUMP_PATH|Path for storing expert hotspot information for load balancing.|All values|None|
|REMOVE_GENERATION_CONFIG_DICT|After this function is enabled, the model postprocessing parameters are set to the default values (valid only for LLMs).|`1`: enabled<br>Other values or `None`: disabled|None|

For details about log-related environment variables, see **Table 5**.

**Table 5** Description of log-related environment variables

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|MINDIE_LOG_LEVEL|Log level.|DEBUG<br>INFO<br>WARN<br>ERROR<br>CRITICAL|INFO|
|MINDIE_LOG_PATH|Path for storing logs.|N/A|"mindie/log/debug"|
|MINDIE_LOG_ROTATE|Size and number of logs to be rotated.|<li>`-fs`: Max size of each log file (MB). Range: [1, 500]</li><li>`-r`: Max number of log files per process. Range: [1, 64]</li>Example: `export MINDIE_LOG_ROTATE="-fs 40 -r 2"`|<li>`-fs`: `20`</li><li>`-r`: `10`</li><br>`PYTHON_LOG_MAXSIZE` takes precedence over the `-fs` parameter in `MINDIE_LOG_ROTATE` when both are set.|
|MINDIE_LOG_TO_FILE|Whether to save logs to files. The value `1` indicates that logs are saved to files.|{0, 1, true, false}|true|
|MINDIE_LOG_TO_STDOUT|Whether to print logs. The value `1` indicates that logs are printed.|{0, 1, true, false}|false|
|MINDIE_LOG_VERBOSE|Whether to add optional content to logs.|{0, 1, true, false}|true|
|PYTHON_LOG_MAXSIZE|Maximum size of a single ATB Python log file (unit: Byte).|[0, 524288000]|None|

For details about the environment variables related to the acceleration library, see **Table 6**.

**Table 6** Environment variables related to the acceleration library

|Environment Variable|Description|Value Range|Default Value|
|--|--|--|--|
|ASCEND_LAUNCH_BLOCKING|Whether to enable synchronous operator delivery, which is used in the debugging scenario.|`0`: disabled<br>`1`: enabled|0|
|ASCEND_RT_VISIBLE_DEVICES|Device ID.|[0, *Device ID*]<br>Example: [0, 1, 2, ...]|N/A|
|ATB_HOME_PATH|Environment variable of the ATB path. There is no default value, and this parameter is required.|N/A|N/A|
|ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT|Number of slots for the global kernel cache. If the number of slots is increased, the cache hit ratio increases, but the retrieval efficiency decreases. If the number of slots is reduced, the retrieval efficiency increases, but the cache hit ratio decreases.|[1, 1024]|16|
|ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT|Number of slots for the local kernel cache. If the number of slots is increased, the cache hit ratio increases, but the retrieval efficiency decreases. If the number of slots is reduced, the retrieval efficiency increases, but the cache hit ratio decreases.|[1, 1024]|1|
|ATB_WORKSPACE_MEM_ALLOC_GLOBAL|Whether to use the global intermediate tensor memory allocation algorithm. After this algorithm is used, the size of the intermediate tensor memory is computed and allocated.|`0`: disabled<br>`1`: enabled|1|

For more ATB-related environment variables, see "Environment Variables" in the *CANN ATB Development Guide*.

> [!NOTE]
>
>- For more PyTorch environment variables such as `INF_NAN_MODE_ENABLE`, `TASK_QUEUE_ENABLE`, and `RANK_TABLE_FILE`, see "INF_NAN_MODE_ENABLE" in *Environment Variable Reference*.
>- When `BIND_CPU` is enabled, `execute_command` is called to run the following commands:
>- `execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "memory"]).split("\n")[1:]` - `execute_command(["npu-smi", "info", "-i", f"{npu_id}", "-t", "usages"]).split("\n")[1:]` - `execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]` - `execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}", "-c", f"{device_info.chip_id}"]).strip().split("\n")` - `execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).split("\n")` - `execute_command(["lscpu"]).split("\n")`
