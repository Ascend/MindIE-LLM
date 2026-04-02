# Splitfuse运行方法

## 1. 当前支持模型
llama3.1-70B

## 2. 使用run_generator.py运行
### 2.1 执行脚本代码示例
    export PYTHONPATH=/LLM代码路径/examples/atb_models/:$PYTHONPATH
    export PYTHONPATH=/LLM代码路径/:$PYTHONPATH

    # 设置执行时运行在哪些NPU上
    export ASCEND_RT_VISIBLE_DEVICES = 0,1

    export INF_NAN_MODE_ENABLE=0

    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1

    export LCCL_DETERMINISTIC=0
    export HCCL_DETERMINISTIC=false
    export ATB_MATMUL_SHUFFLE_K_ENABLE=1
    export ATB_LLM_LCOC_ENABLE=1
    
    # LOG开关按需设置（可按需调整）
    export MINDIE_LOG_LEVEL=INFO
    export MINDIE_LOG_TO_FILE=1
    export MINDIE_LOG_TO_STDOUT=1

    # 1表示执行的NPU数目，可按需调整
    torchrun \
    --nproc_per_node 1 \
    --master_port 23333 \
    -m mindie_llm.examples.run_generator \
    --model_path /模型路径/ \
    --max_batch_size 2 \
    --max_output_length 128 \
    --plugin_params "{\"plugin_type\":\"splitfuse\"}" \
    --split_chunk_tokens 10

### 2.2 运行
1. 编译：代码根目录下执行 bash scripts/build.sh
2. source output/set_env.sh
3. 执行2.1中的脚本。

## 3. 连接服务化执行

### 3.1 对话验证 mindieservice_daemon
#### 3.1.1 config文件配置
此处只呈现启动Splitfuse需要额外配置的字段

"ModelDeployConfig"中的"ModelParam"下添加：

    "plugin_params": "{\"plugin_type\":\"splitfuse\"}" 
    
"ScheduleConfig"中修改"templateType"为"Mix"
"ScheduleConfig"中添加：

    "policyType": 0,
    "enableSplit": true,
    "splitType": true,
    "splitStartType": false,
    "splitChunkTokens": 512,
    "splitStartBatchSize": 16

[说明]<br>
policyType: 可配置0，4，5，6，7<br>

#### 3.1.2 执行
1. 执行./mindieservice_daemon
2. 执行对话请求即可运行<br>