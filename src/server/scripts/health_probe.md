# 健康探针脚本 (health_probe.sh) 与兜底方案

## 一、背景

**健康探针脚本（health_probe.sh）** 适用于检测 **PD 混部场景** 下服务是否正常的示例脚本。  
如果服务化出现异常（静默故障如超时，非静默故障如 coredump），可使用以下兜底方案进行止血。

本文分为两部分：

1. 对健康探针脚本 `health_probe.sh` 进行代码解释，帮助熟悉脚本原理。  
2. 基于该脚本实现用户反馈所需的 **兜底方案**。由于用户反馈场景各异，需根据具体诉求（如：新增日志、增加故障处理逻辑等）进行定制。

> ⚠️ 注意：`health_probe.sh` 属于示例脚本，**不能直接一键部署**。  
> 请根据实际场景编写适配用户诉求的一键部署脚本。

---

## 二、代码解读

### 1. 使用健康探针示例（必选）

```bash
#######################################################################################
# Check /health/timed-3
#######################################################################################

# 读取 PD 混部服务化配置参数中的管理面端口
config_file="/home/HwHiAiUser/Ascend/mindie/latest/mindie-service/conf/config.json"
management_port=$(grep '"managementPort"' "$config_file" | sed 's/[^0-9]*//g')

# 调用 PD 混部健康探针接口 /health/timed-${TIMEOUT}
# 详情请参考：https://www.hiascend.com/document/detail/zh/mindie/21RC1/mindieservice/servicedev/mindie_service0102.html
response_file=~/health_response
curl --silent --write-out "HTTPSTATUS:%{http_code}" -m 3 \
     "http://xx.xx.xx.xx:$management_port/health/timed-3" > "$response_file" &
```

---

### 2. 监控 AICore 示例（可选）

```bash
#######################################################################################
# Check npu-smi info
#######################################################################################

npu_id=$(awk 'NR==2 {print $1}' ~/device_info)
max_aicore_usage=0
num_samples=4

for ((i=0; i<num_samples; i++)); do
    output=$(npu-smi info -t usages -i "$npu_id")
    aicore_usage=$(echo "$output" | grep 'Aicore Usage Rate(%)' | awk '{print $NF}' | tr -d '%')
    if [[ -n "$aicore_usage" && "$aicore_usage" -gt "$max_aicore_usage" ]]; then
        max_aicore_usage=$aicore_usage
    fi
    if (( i < num_samples - 1 )); then
        sleep 0.1
    fi
done

if (( max_aicore_usage < 10 )); then
    core_abnormal=true
else
    core_abnormal=false
fi
```

---

### 3. 基于监控结果的最终判断示例

```bash
#######################################################################################
# Final conclusion
#######################################################################################

wait $!

response=$(<"$response_file")
response_body=$(echo "$response" | sed -e 's/HTTPSTATUS\:.*//g')
response_code=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')

if [[ "$response_code" -ne 200 ]] || [[ "$response_body" != '{"status":"healthy"}' ]]; then
    timed_out=true
else
    timed_out=false
fi

if [[ "$timed_out" == true && "$core_abnormal" == true ]]; then
    max_aicore_usage=0
    num_samples=6

    for ((i=0; i<num_samples; i++)); do
        output=$(npu-smi info -t usages -i "$npu_id")
        aicore_usage=$(echo "$output" | grep 'Aicore Usage Rate(%)' | awk '{print $NF}' | tr -d '%')
        if [[ -n "$aicore_usage" && "$aicore_usage" -gt "$max_aicore_usage" ]]; then
            max_aicore_usage=$aicore_usage
        fi
        if (( i < num_samples - 1 )); then
            sleep 0.1
        fi
    done

    if (( max_aicore_usage < 10 )); then
        core_abnormal=true
    else
        core_abnormal=false
    fi

    if [[ "$timed_out" == true && "$core_abnormal" == true ]]; then
        echo 501
    else
        echo 200
    fi
else
    echo 200
fi
```

---

## 三、兜底方案实现

在实现兜底方案前，请先判断用户的服务是否存在 **空转可能**。  
如果存在空转，则 **无需监控 AICore**。

### 兜底逻辑（伪代码）

```bash
while true; do
    if check_service_health; then
        # 服务正常
    else
        # 服务异常，重新启动服务
        restart_service
    fi
    sleep 60
done
```

---

### 健康检查函数实现

```bash
# 函数：check_service_health
# 功能：检查服务健康状态，支持最多三次重试
# 返回值：
#   0 -> 服务健康
#   1 -> 服务异常或重试多次仍未健康
check_service_health() {
    local response
    local http_code
    local content
    local retry_count=0
    local max_retries=3

    while [ $retry_count -lt $max_retries ]; do
        response=$(curl -s -w "\n%{http_code}" "$HEALTH_CHECK_URL" 2>/dev/null)
        http_code=$(echo "$response" | tail -n1)
        content=$(echo "$response" | sed '$d')

        if [ "$http_code" = "200" ] && [ "$content" = '{"status":"healthy"}' ]; then
            return 0
        fi

        retry_count=$((retry_count + 1))
        sleep 1
    done

    return 1
}
```

---

## 四、完整兜底脚本示例

以下脚本每 **5 秒检查一次服务健康状态**，  
每次检查包含 **3 次重试（间隔 1 秒）**。  
若连续失败，则触发服务重启。  
若服务启动后持续异常超过 **15 分钟（900 秒）**，将再次自动重启。

```bash
#!/bin/bash

# 日志文件
LOG_FILE="monitor.log"

# 状态定义
STATUS_INIT="INIT"
STATUS_NORMAL="NORMAL"
STATUS_START="START"

# 服务健康检查 URL
HEALTH_CHECK_URL="http://xx.xx.xx.xx:xxxx/health/timed"

# 初始状态
current_status="$STATUS_INIT"
start_time=0

# 日志函数
log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

# 健康检查函数
check_service_health() {
    local response
    local http_code
    local retry_count=0
    local max_retries=3

    while [ $retry_count -lt $max_retries ]; do
        response=$(curl -s -w "\n%{http_code}" "$HEALTH_CHECK_URL" 2>/dev/null)
        http_code=$(echo "$response" | tail -n1)
        content=$(echo "$response" | sed '$d')

        if [ "$http_code" = "200" ] && [ "$content" = '{"status":"healthy"}' ]; then
            return 0
        fi

        retry_count=$((retry_count + 1))
        sleep 1
    done

    return 1
}

# 服务重启函数
restart_service() {
    log_message "开始重启服务..."

    pkill -9 mindie
    log_message "已发送终止 mindie 进程信号"

    pkill -9 python
    log_message "已发送终止 python 进程信号"

    local wait_count=0
    local max_wait=10

    while [ $wait_count -lt $max_wait ]; do
        pkill -9 mindie
        pkill -9 python
        if pgrep -x "mindie" > /dev/null || pgrep -f "python" > /dev/null; then
            sleep 1
            wait_count=$((wait_count + 1))
        else
            break
        fi
    done

    if pgrep -x "mindie" > /dev/null; then
        log_message "警告: mindie 进程仍然存在"
    fi

    if pgrep -f "python" > /dev/null; then
        log_message "警告: python 进程仍然存在"
    fi

    log_message "正在启动服务..."
    bash A2_single_machine.sh
    log_message "服务启动命令已执行"

    current_status="$STATUS_START"
    start_time=$(date +%s)
    log_message "状态变更为: $STATUS_START"
}

# 主循环
log_message "监控脚本启动，初始状态: $STATUS_INIT"

while true; do
    if check_service_health; then
        case "$current_status" in
            "$STATUS_INIT")
                log_message "服务健康，状态从 $STATUS_INIT 变更为 $STATUS_NORMAL"
                current_status="$STATUS_NORMAL"
                ;;
            "$STATUS_START")
                log_message "服务健康，状态从 $STATUS_START 变更为 $STATUS_NORMAL"
                current_status="$STATUS_NORMAL"
                ;;
        esac
    else
        case "$current_status" in
            "$STATUS_INIT")
                log_message "服务异常，状态从 $STATUS_INIT 变更为 $STATUS_START"
                current_status="$STATUS_START"
                start_time=$(date +%s)
                ;;
            "$STATUS_NORMAL")
                log_message "检测到服务异常，状态从 $STATUS_NORMAL 变更为 $STATUS_START"
                current_status="$STATUS_START"
                start_time=$(date +%s)
                restart_service
                ;;
            "$STATUS_START")
                current_time=$(date +%s)
                elapsed_time=$((current_time - start_time))

                if [ $elapsed_time -ge 900 ]; then
                    log_message "服务异常已持续15分钟，重新启动服务..."
                    restart_service
                else
                    log_message "服务仍然异常，已持续 ${elapsed_time} 秒"
                fi
                ;;
        esac
    fi

    sleep 5
done
```

---

## 五、总结

该兜底脚本具备以下功能：

- ✅ 每 5 秒自动检查服务健康状态  
- ✅ 每次最多重试 3 次（每次间隔 1 秒）  
- ✅ 检测到异常自动重启服务  
- ✅ 异常持续 15 分钟会强制重启  
- ✅ 日志记录完整状态变化与操作过程  

该脚本可根据用户诉求扩展，如：
- 增加日志上传到远端；
- 增加多进程服务重启逻辑；
- 结合 NPU 使用率进行更细粒度判定。
