# Quick Start

## Environment Setup

This document uses the Atlas 800I A2 inference server and Qwen2-7B model as examples to help developers quickly get started with foundation model inference using MindIE.

### Prerequisites

Install NPU driver and firmware and deploy Docker on a physical machine. You can perform the following steps to check whether the required software is installed or deployed.

- Check whether the NPU driver and firmware are installed. If information similar to that shown in [Figure 1](#figure1) is displayed, the NPU driver and firmware have been installed. Otherwise, install them by referring to [Table 1](#table1).

     ```bash
     npu-smi info
     ```

    **Figure 1** Command output<a id="figure1"></a>

    ![](./figures/command_output.png "Command output")

    **Table 1** Atlas A2 inference products<a id="table1"></a>

    |Product Model |Reference Document|
    |------------|------------|
    | Atlas 800I A2 | Download the [firmware and driver](https://hiascend.com/hardware/firmware-drivers/community). For details about how to install the firmware and driver, see "[Installing the NPU Driver and Firmware](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler)" in CANN Software Installation (commercial edition). |

- Check whether Docker has been installed and started. For details about how to install Docker, see [Installing Docker](../install/source/docker_installation.md).

     ```bash
     docker ps
     ```

    If the following information is displayed, the Docker has been installed and started.

     ```text
     CONTAINER ID        IMAGE        COMMAND         CREATED        STATUS         PORTS           NAMES
     ```

### Downloading Model Weights

1. Download the weight file first. The following uses Qwen2-7B as an example. Download its weight file from [https://huggingface.co/Qwen/Qwen2-7B/tree/main](https://huggingface.co/Qwen/Qwen2-7B/tree/main) and upload it to any directory, for example, `/home/weight`, on the server.
2. Run the following command to change the permission on the weight file:

     ```bash
     chmod -R 755 /home/weight
     ```

### Obtaining the Container Image

Go to the [Ascend hub](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f) and download the corresponding MindIE image based on the device model.

This image comes pre-configured with the base environment required for model execution, including CANN, FrameworkPTAdapter, MindIE, and ATB Models, enabling rapid inference setup.

**Table 2** Installation path of each component in a container

|Component|Installation Path|
|--|--|
|CANN|/usr/local/Ascend/cann|
|CANN-NNAL-ATB|/usr/local/Ascend/nnal/atb|
|MindIE|/usr/local/Ascend/mindie|
|ATB Models|/usr/local/Ascend/atb-models|

## Starting the Container

1. Start the container after the image is downloaded.

    ```bash
    docker run -it -d --net=host --shm-size=1g \
           --name <container-name> \
           -w /home \
           --device=/dev/davinci0:rwm \
           --device=/dev/davinci1:rwm \
           --device=/dev/davinci2:rwm \
           --device=/dev/davinci3:rwm \
           --device=/dev/davinci_manager:rwm \
           --device=/dev/hisi_hdc:rwm \
           --device=/dev/devmm_svm:rwm \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
           -v /usr/local/dcmi:/usr/local/dcmi:ro \
           -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
           -v /usr/local/sbin/:/usr/local/sbin:ro \
           -v /home/weight:/home/weight:ro \
           mindie:3.0.0-800I-A2-py311-openeuler24.03-lts bash
    ```

     > [!NOTE]NOTE
     > - `mindie:3.0.0-800I-A2-py311-openeuler24.03-lts` is the image name, which can be changed as required.
     > - For the `--device` parameter, the mount permission is set to `rwm` instead of the less permissive `rw` or `r`, for the following reasons:
     > - For Atlas 800I A2 inference servers, setting the mount permission to `rw` allows normal container entry. The `npu-smi` command can still display NPU usage info, and MindIE workloads run correctly. However, if the mounted NPU (e.g., `davinci0` for `npu0`) is already occupied by other tasks, `npu-smi` will print an error, and MindIE tasks will fail (e.g., `torch.npu.set_device()` will fail).
     > - For Atlas 800I A3 SuperPoD servers, setting the mount permission to `rw` may cause `npu-smi` to return an error inside the container, and MindIE tasks will fail (e.g., `torch.npu.set_device()` will fail).

    **Table 1** Parameter description

    |Parameter|Description|
    |--|--|
    |-it|Starts an interactive terminal (-i) and connects it to the standard input and output of the container (-t). In this way, the terminal can interact with the container, for example, running commands.|
    |-d|Indicates that the container runs in the background. That is, the container is started in the background. After this parameter is used, the operations on the current terminal are not blocked. You can perform other operations after the container is started.|
    |--net|Indicates that the container uses the network configuration (network sharing) of the host so that the container can directly access the network interface of the host. This parameter applies to scenarios where low latency and direct access to network resources are required.|
    |--shm-size|Specifies the size of the shared memory (**/dev/shm**) of a specified container. You can set the size as required. 1g is an example value. For multimodal understanding models, if the maximum service concurrency is high, setting this parameter to at least 100 GB is recommended.<br>The value cannot exceed the size of the remaining physical memory of the host. You can run the `free -h` command to view the size of the remaining physical memory. When data parallelism (DP) is enabled, the shared memory size (**shm-size**) must be adjusted proportionally as the DP value grows beyond 1.<ul><li>For a DP value of 2, set **shm-size** to at least 2 GB.</li><li>For a DP value of 4, set **shm-size** to at least 3 GB.</li><li>For a DP value of 8, set **shm-size** to at least 5 GB.</li><li>For a DP value of 16, set **shm-size** to at least 9 GB.</li></ul>|
    |--name|Specifies a name for the container. <*container-name*> is the identifier of the container, which can be customized and must be unique in the current system. If this parameter is not set, Docker automatically allocates a random name.|
    |--device|Indicates the mapped device. One or more devices can be mounted.<br>The devices to be mounted are as follows: <ul><li>`/dev/davinciX`: NPU device, where `X` is the ID number (e.g., `davinci0`). </li><li>`/dev/davinci_manager`: Management device for davinci. </li><li>`/dev/hisi_hdc`: Management device for hdc. </li><li>/`/dev/devmm_svm`: Memory management device. </li></ul>Run `ll /dev/ \| grep davinci` to check the number and names of available devices. Bind the required devices by modifying `--device=****` in the command above.|
    |-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro|Mounts the host directory `/usr/local/Ascend/driver` to the container. Change it according to the actual driver path.|
    |-v /usr/local/sbin:/usr/local/sbin:ro|Mounts the host tool `/usr/local/sbin/` to the container in read-only mode. Change it as required.|
    |-v /home/weight:/home/weight:ro|Sets the weight mounting path as required. Place the weight file and dataset file in the same path.|

2. Access the container.

     ```bash
     docker exec -it <container-name> /bin/bash
     ```

     > [!NOTE]NOTE
     > For details, see [Starting a Container](https://gitee.com/ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8).

## Model Inference

1. If the default installation path is used, run the following command to go to the MindIE installation directory::

     ```bash
     cd /usr/local/Ascend/mindie/latest
     ```

2. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.:

     ```bash
    chmod 750 mindie-service
    chmod -R 550 mindie-service/bin
    chmod 550 mindie-service/lib
    chmod 440 mindie-service/lib/*
    chmod 550 mindie-service/lib/grpc
    chmod 440 mindie-service/lib/grpc/*
    chmod -R 550 mindie-service/include
    chmod -R 550 mindie-service/scripts
    chmod 750 mindie-service/logs
    chmod 750 mindie-service/conf
    chmod 640 mindie-service/conf/config.json
    chmod 700 mindie-service/security
    chmod -R 700 mindie-service/security/*
     ```

     > [!NOTE]NOTE
     > If the file permission does not meet the requirements, the service will fail to be started.

3. Set environment variables. <a id="step3"></a>
   Run the following commands to initialize the environment variables of each component and enable log printing

     ```bash
    # Configure the CANN environment. By default, the CANN is installed in the /usr/local directory.
    source /usr/local/Ascend/cann/set_env.sh
    # Configure the acceleration library environment variables.
    source /usr/local/Ascend/nnal/atb/set_env.sh
    # Configure the model repository environment variables.
    source /usr/local/Ascend/atb-models/set_env.sh
    # MindIE
    source /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
    source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
    # Enable MindIE log printing.
    export MINDIE_LOG_TO_STDOUT="true"
     ```

4. Set serving parameters.

    a. Go to the `conf` directory and open the `config.json` file.

    ```bash
    cd mindie-service/conf
    vim config.json
    ```

    b. Press **i** to enter insert mode and modify the parameters in the `config.json` file as required. (The following uses Qwen2-7B as an example. The parameters to be modified are in bold.)

    ``` json

     {
        "ServerConfig" :
            {
            "httpsEnabled" : false
            },
        "BackendConfig" :
         {
                "npuDeviceIds" : [[0,1,2,3]],
                "ModelDeployConfig" :
            {
                    "ModelConfig" : [
                    {
                        "modelName" : "qwen2-7b",
                        "modelWeightPath" : "/home/weight",
                        "worldSize" : 4,
                        "trustRemoteCode": false
                    }
                ]
            },
        }
    }
    ```

    The preceding parameters are described as follows. For details about the parameters in `config.json`, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md).

    |Parameter|Value Type|Value Range|Configuration Description|
    |--|--|--|--|
    |httpsEnabled|bool|`true` (enabled) and `false` (disabled)|Whether to enable HTTPS communication security authentication. `true`: enables HTTPS communication. `false`: disables HTTPS communication. If the network environment is insecure, HTTPS communication will be disabled (`"httpsEnabled" : false`) due to high network security risks.|
    |npuDeviceIds|std::vector<std::set<size_t>>|Set this parameter based on the model and environment.|NPUs to be enabled. The NPU ID allocated to each model instance is represented by the logical processor ID. If `ASCEND_RT_VISIBLE_DEVICES` is not configured, you can run the `npu-smi info -m` command to query the logical ID of each device. If `ASCEND_RT_VISIBLE_DEVICES` is configured, the logical IDs of visible devices start from 0 based on the sequence configured in `ASCEND_RT_VISIBLE_DEVICES`. For example, if `ASCEND_RT_VISIBLE_DEVICES` is set to `1,2,3,4`, the logical IDs of visible devices are 0, 1, 2, and 3 in sequence. This parameter is invalid in multi-server inference scenarios. The value of `npuDeviceIds` used on each node is calculated based on `ranktable`. This parameter is required. The default value is `[[0,1,2,3]]`.|
    |modelName|string|The value can contain a maximum of 256 characters, including uppercase letters, lowercase letters, digits, hyphens (-), periods (.), and underscores (_). It cannot start or end with a hyphen (-), period (.), or underscore (_).|Model name. This parameter is required. The default value is `llama_65b`.|
    |modelWeightPath|std::string|The maximum length of an absolute file path depends on the setting of the operating system (`PATH_MAX` in Linux). The minimum value is `1`.|Path of the model weight file. The program reads the values of the `torch_dtype` and `vocab_size` fields in the `config.json` file in the path. Ensure that the path and related fields exist. This parameter is required. The default value is `/data/atb_testdata/weights/llama1-65b-safetensors`. Security verification is performed on the path. The owner group and permission of the path must be the same as those of the execution user.|
    |worldSize|uint32_t|Set this parameter based on the actual situation of the model. The value of `worldSize` in each set of model parameters must be the same as the number of NPUs in use.|Number of NPUs used for inference. This parameter is required. The default value is `4`.|
    |trustRemoteCode|bool|`true` or `false`|Whether to trust remote code. `false`: Remote code is not trusted. `true`: Remote code is trusted. This parameter is optional. The default value is `false`. If this parameter is set to `true`, remote code is trusted, which may cause malicious code injection risks. You need to guarantee code injection security.|

    c. Press `Esc`, type `:wq!`, and press `Enter` to save the settings and exit.

5. Start the service.

    a. Run the following command to go to the installation directory::

    ```bash
    cd /usr/local/Ascend/mindie/latest/mindie-service
    ```

    b. Start the service using either of the following methods:

    - Method 1 (recommended): Start the service using a background process. After the service is started in background process mode, the process is retained when the window is closed.
  
    ```bash
    nohup ./bin/mindieservice_daemon > output.log 2>&1 &
    ```

    If the following information is printed in the file captured by the standard output stream, the startup is successful:

    ```text
    Daemon start success!
    ```

    - Method 2: Directly start the service.
  
    ```bash
    ./bin/mindieservice_daemon
    ```

    If the following information is displayed, the service is started successfully:

    ```text
    Daemon start success!
    ```

     > [!CAUTION]NOTE
     >- To avoid conflicts with an earlier version of MindIE (default install path: `/usr/local/Ascend/mindie`), run the `mv /usr/local/Ascend/mindie /usr/local/Ascend/mindie-bak` command. This removes files in the old installation path and prevents the system from linking to outdated libraries.
     >- To meet security requirements, the `bin` directory is set with `550` permissions (no write access). However, during inference execution, operators need to generate a `kernel_meta` folder in the current directory, which requires write permission. Therefore, `mindieservice_daemon` cannot be started directly in the `bin` directory.
     >- When Ascend-CANN-Toolkit is used, the inference service generates a `kernel_meta_temp_xxxx` directory in the launch directory to store operator `.cce` files. Therefore, the service must be started in a directory where the current user has write permissions, such as `Ascend-mindie-server_{version}_linux-{arch}`, or a user-created temporary directory under it.
     >- To switch to another user, run the `rm -f /dev/shm/*` command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
     >- The `output.log` file captured by the standard output stream supports user-defined files and paths.

6. Send a request.

    For details about serving APIs, see "RESTful API Reference" in *MindIE LLM Development Guide*.

    You can use an HTTPS client (Linux `curl` command, Postman, and others) to send HTTPS requests. The following uses Linux **curl** command as an example.

    Open a new window and send a request, for example, to check whether the verification service is started.

    ```bash
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "prompt": "My name is Olivier and I ",
    "max_tokens":10
    }' http://127.0.0.1:1025/generate
    ```

    If the following information is displayed, the request is successfully sent:

    ```text
    {"text":["My name is Olivier and I  25 years old. I am a French student"]}
    ```

## Accuracy Test

> [!NOTE]NOTE
>
>- Before the accuracy and performance tests, open another window to access the container and set environment variables by referring to [3](#step3).
>- The following uses the AISBench tool as an example to describe the accuracy test. For details about how to use the AISBench tool, see [AISBench](https://gitee.com/aisbench/benchmark).

1. Download and install AISBench.

    ```bash
    git clone https://gitee.com/aisbench/benchmark.git
    cd benchmark/
    pip3 install -e ./ --use-pep517
    pip3 install -r requirements/api.txt
    pip3 install -r requirements/extra.txt
    ```

    > [!NOTE]NOTE
    > The `pip` installation mode applies to scenarios where the latest functions of AISBench are used (except the scenario where MindIE is installed using an image). AISBench has been pre-installed in the MindIE image. You can run the following command to view the installation path of AISBench in the MindIE image:
        >
        >```bash
        >pip show ais_bench_benchmark
        >```

2. Prepare a dataset.

    Using gsm8k as an example: download the dataset by clicking [gsm8k](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip), then extract the archive and place the `gsm8k` folder under `ais_bench/datasets` in the tool root directory.

3. Configure the `ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` file. The following is an example:

    ```python
    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='vllm-api-stream-chat',
            path="/home/weight",                    # Absolute path to the model serialized vocabulary file, which is generally the path to the model weight folder.
            model="qwen2-7b",        # Name of the model loaded on the server. Set this parameter based on the name of the model pulled by the vLLM inference service. (If this parameter is set to an empty string, the model name is automatically obtained.)
            request_rate = 0,           # Request sending frequency. One request is sent to the server every 1/request_rate second. If the value is less than 0.1, all requests are sent at a time.
            retry = 2,
            host_ip = "127.0.0.1",      # IP address of the inference service
            host_port = 1025,           # Port number of the inference service
            max_out_len = 512,          # Maximum number of tokens output by the inference service
            batch_size=1,               # Maximum number of concurrent requests to be sent
            trust_remote_code=False,
            generation_kwargs = dict(
                temperature = 0.5,
                top_k = 10,
                top_p = 0.95,
                seed = None,
                repetition_penalty = 1.03,
            ) ,
             pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]
    ```

4. Run the following command to start the serving accuracy test:

    ```bash
    ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
    ```

    The command is executed successfully if the command output is as follows:

    ```text
    dataset                 version  metric   mode  vllm_api_general_chat
    ----------------------- -------- -------- ----- ----------------------
    demo_gsm8k              401e4c   accuracy gen                   62.50
    ```

## Performance Test

> [!NOTE]NOTE
> The following uses the AISBench tool as an example to describe the performance test. For details about how to use the AISBench tool, see [AISBench](https://gitee.com/aisbench/benchmark).

1. Download and install AISBench.

    ```bash
    git clone https://gitee.com/aisbench/benchmark.git
    cd benchmark/
    pip3 install -e ./ --use-pep517
    pip3 install -r requirements/api.txt
    pip3 install -r requirements/extra.txt
    ```

    > [!NOTE]NOTE
    > The·`pip` installation mode applies to scenarios where the latest functions of AISBench are used (except the scenario where MindIE is installed using an image). AISBench has been pre-installed in the MindIE image. You can run the following command to view the installation path of AISBench in the MindIE image:
        >
        >```bash
        >pip show ais_bench_benchmark
        >```

2. Prepare a dataset.

    Using gsm8k as an example: download the dataset by clicking [gsm8k](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip), then extract the archive and place the `gsm8k/` folder under `ais_bench/datasets` in the tool root directory.

3. Configure the `ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` file. The following is an example:

    ```python
    from ais_bench.benchmark.models import VLLMCustomAPIChatStream
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChatStream,
            abbr='vllm-api-stream-chat',
            path="/home/weight",                    # Absolute path to the model serialized vocabulary file, which is generally the path to the model weight folder.
            model="qwen2-7b",        # Name of the model loaded on the server. Set this parameter based on the name of the model pulled by the vLLM inference service. (If this parameter is set to an empty string, the model name is automatically obtained.)
            request_rate = 0,           # Request sending frequency. One request is sent to the server every 1/request_rate second. If the value is less than 0.1, all requests are sent at a time.
            retry = 2,
            host_ip = "127.0.0.1",      # IP address of the inference service
            host_port = 1025,           # Port number of the inference service
            max_out_len = 512,          # Maximum number of tokens output by the inference service
            batch_size=1,               # Maximum number of concurrent requests to be sent
            trust_remote_code=False,
            generation_kwargs = dict(
                temperature = 0.5,
                top_k = 10,
                top_p = 0.95,
                seed = None,
                repetition_penalty = 1.03,
                ignore_eos = True,      # The inference service output ignores EOS (the output length reaches max_out_len).
            ) ,
             pred_postprocessor=dict(type=extract_non_reasoning_content)
        )
    ]
    ```

4. Run the following command to start the serving performance test:

    ```bash
    ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --debug
    ```

    The command is executed successfully if the command output is as follows:

    ```text

    │ Performance Parameters │ Stage  │ Average        │ Min          │ Max        │ Median       │ P75        │ P90          │ P99          │ N │
    │ E2EL                   │total   │ 2048.2945  ms  │ 1729.7498 ms │ 3450.96 ms │ 2491.8789 ms │ 2750.85 ms │ 3184.9186 ms │ 3424.4354 ms │ 8 │
    │ TTFT                   │total   │ 50.332 ms      │ 50.6244 ms   │ 52.0585 ms │ 50.3237 ms   │ 50.5872 ms │ 50.7566 ms   │ 50.0551 ms   │ 8 │
    │ TPOT                   │total   │ 10.6965 ms     │ 10.061 ms    │ 10.8805 ms │ 10.7495 ms   │ 10.7818 ms │ 10.808 ms    │ 10.8582 ms   │ 8 │
    │ ITL                    │total   │ 10.6965 ms     │ 7.3583 ms    │ 13.7707 ms │ 10.7513 ms   │ 10.8009 ms │ 10.8358 ms   │ 10.9322 ms   │ 8 │
    │ InputTokens            │total   │ 1512.5         │ 1481.0       │ 1566.0     │ 1511.5       │ 1520.25    │ 1536.6       │ 1563.06      │ 8 │
    │ OutputTokens           │total   │ 287.375        │ 200.0        │ 407.0      │ 280.0        │ 322.75     │ 374.8        │ 403.78       │ 8 │
    │ OutputTokenThroughput  │total   │ 115.9216       │ 107.6555     │ 116.5352   │ 117.6448     │ 118.2426   │ 118.3765     │ 118.6388     │ 8 │

    ```

    ```text

    │ Common Metric            │ Stage    │ Value              │
    │ Benchmark Duration       │ total    │ 19897.8505 ms      │
    │ Total Requests           │ total    │ 8                  │
    │ Failed Requests          │ total    │ 0                  │
    │ Success Requests         │ total    │ 8                  │
    │ Concurrency              │ total    │ 0.9972             │
    │ Max Concurrency          │ total    │ 1                  │
    │ Request Throughput       │ total    │ 0.4021 req/s       │
    │ Total Input Tokens       │ total    │ 12100              │
    │ Prefill Token Throughput │ total    │ 17014.3123 token/s │
    │ Total generated tokens   │ total    │ 2299               │
    │ Input Token Throughput   │ total    │ 608.7438 token/s   │
    │ Output Token Throughput  │ total    │ 115.7835 token/s   │
    │ Total Token Throughput   │ total    │ 723.5273 token/s   │


    ```

    Performance test results focus primarily on TTFT, TPOT, Request Throughput, and Output Token Throughput. For detailed parameter descriptions, refer to "Table 2: Performance Test Result Metric Comparison" in "Auxiliary Tools" > "Performance/Accuracy Test Tools" in *MindIE Motor Development Guide*.

    > [!NOTE]NOTE
    > The task execution result is ultimately written to the default output directory, as indicated in the runtime log:
        >
        >```text
        > 08/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250828_151326
        >```
    >
    > After the command is executed, the task execution details in `outputs/default/20250828_151326` are as follows:
        >
        >```text
        > 20250828_151326           # Unique directory generated based on the timestamp for each experiment
        >├── configs               # All dumped configuration files that are automatically stored
        >├── logs                  # Logs generated during the execution. If --debug is added to the command, no process logs will be written to drive (they will be printed directly to the console).
        >│   └── performance/      # Log files for the inference phase
        >└── performance           # Performance evaluation result
        >│    └── vllm-api-stream-chat/          # Serving model configuration name, which corresponds to the abbr parameter of models in the model task configuration file
        >│         ├── gsm8kdataset.csv          # Per-request performance output (CSV), which matches the "Performance Parameters" table printed in the results
        >│         ├── gsm8kdataset.json         # End-to-end performance output (JSON), which matches the "Common Metrics" table printed in the results
        >│         ├── gsm8kdataset_details.json # Full trace event log (JSON)
        >│         └── gsm8kdataset_plot.html    # Visualized report of concurrent requests (HTML)
        >```
