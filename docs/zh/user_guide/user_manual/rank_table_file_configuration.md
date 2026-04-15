# rank_table_file.json 配置指南

## 概述

在多机推理场景下，MindIE-LLM 依赖 HCCL（Huawei Collective Communication Library）进行多节点间的集合通信。为了让各节点上的 NPU 卡能够互相发现并建立通信连接，需要用户提供一个 `rank_table_file.json` 文件，用于描述集群中每张卡的拓扑信息。

`rank_table_file.json` 本质上是一份“通信拓扑表”，它告诉 HCCL 以下几个关键信息：

- 集群中有多少台机器（server_count）
- 每台机器有哪些 NPU 卡（device_id）
- 每张 NPU 卡的网络 IP 地址（device_ip）
- 每张 NPU 卡在全局中的编号（rank_id）
- 每台机器的网络 IP 地址（server_id）

本文档将逐步介绍如何完成网络检查、获取 IP 地址、编写 `rank_table_file.json` 文件，以及相关的权限配置。

---

## 前提条件

- 已完成 MindIE-LLM 的安装，参见[安装指南](../install/installing_MindIE.md)。
- 已确认集群中各节点的 NPU 卡状态正常，可通过 `npu-smi info` 命令检查。
- 多机场景下，各节点间网络互通。

---

## 操作步骤

### 步骤一：检查机器网络情况

在配置 `rank_table_file.json` 之前，建议先检查各节点的 NPU 网络状态，确保物理链路连通、网络健康、网关和 TLS 配置正确。以下命令需要在每台机器上执行（以 A3 环境、每机 16 张 NPU 卡为例）。

> [!NOTE]
>
> 以下命令均使用 `hccn_tool` 工具，该工具随 CANN 安装包提供。如果命令不可用，请检查 CANN 环境是否正确配置。

1. 检查物理链路连通性：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
    ```

2. 检查链路状态：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -link -g; done
    ```

3. 检查网络健康情况：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -net_health -g; done
    ```

4. 检查侦测 IP 配置是否正确：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -netdetect -g; done
    ```

5. 检查网关配置是否正确：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -gateway -g; done
    ```

6. 检查 NPU 底层 TLS 校验行为一致性：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -tls -g; done | grep switch
    ```

7. 将 NPU 底层 TLS 校验行为统一设置为 0：

    ```shell
    for i in {0..15}; do hccn_tool -i $i -tls -s enable 0; done
    ```

> [!NOTE]
>
> 建议统一将所有卡的 TLS 校验行为设置为 0，避免 HCCL 报错。

---

### 步骤二：获取每张 NPU 卡的 IP 地址

在每台机器上执行以下命令，获取该机器上每张 NPU 卡对应的 IP 地址：

```shell
for i in {0..15}; do hccn_tool -i $i -ip -g; done
```

执行后会输出每张卡的 IP 地址信息，请将结果记录下来，后续编写 `rank_table_file.json` 时需要使用。

> [!NOTE]
>
> - 每张 NPU 卡都有一个独立的 RoCE（RDMA over Converged Ethernet）网口，拥有自己的 IP 地址，此 IP 地址用于卡间高速通信。
> - 不同机器上的 NPU 卡数可能不同，请根据实际情况调整循环范围。例如，每机 8 卡时使用 `{0..7}`。

---

### 步骤三：编写 rank_table_file.json

#### 文件格式说明

`rank_table_file.json` 是一个 JSON 格式的配置文件，包含集群中所有节点和所有 NPU 卡的拓扑信息。

**表 1** `rank_table_file.json` 参数说明

| 参数 | 说明 |
|------|------|
| server_count | 总节点数（即机器数量）。 |
| server_list | 节点列表，其中第一个 server 为主节点（Master）。 |
| device_id | 当前卡在本机内的编号，取值范围为 [0, 本机卡数)。 |
| device_ip | 当前卡的 IP 地址，通过 `hccn_tool` 命令获取。 |
| rank_id | 当前卡在全局（所有机器的所有卡）中的唯一编号，取值范围为 [0, 总卡数)。 |
| server_id | 当前节点的 IP 地址（即机器的网络 IP）。 |
| container_ip | 容器 IP 地址（服务化部署时需要）。若无特殊配置，则与 server_id 相同。 |
| status | 固定填写 `"completed"`。 |
| version | 固定填写 `"1.0"`。 |

#### rank_id 编号规则

`rank_id` 是每张 NPU 卡在全局中的唯一编号，从 0 开始递增。编号规则为按节点顺序、按卡号顺序依次分配。

例如，双机各 16 卡的场景：

- 第 1 台机器：卡 0~15 的 rank_id 分别为 0~15
- 第 2 台机器：卡 0~15 的 rank_id 分别为 16~31

#### 双机样例

以下是一个双机、每机 16 张 NPU 卡的完整样例。用户需要将 `device_ip`、`server_id`、`container_ip` 字段替换为实际的 IP 地址。

```json
{
   "server_count": "2",
   "server_list": [
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "192.168.1.10",
               "rank_id": "0"
            },
            {
               "device_id": "1",
               "device_ip": "192.168.1.11",
               "rank_id": "1"
            },
            {
               "device_id": "2",
               "device_ip": "192.168.1.12",
               "rank_id": "2"
            },
            {
               "device_id": "3",
               "device_ip": "192.168.1.13",
               "rank_id": "3"
            },
            {
               "device_id": "4",
               "device_ip": "192.168.1.14",
               "rank_id": "4"
            },
            {
               "device_id": "5",
               "device_ip": "192.168.1.15",
               "rank_id": "5"
            },
            {
               "device_id": "6",
               "device_ip": "192.168.1.16",
               "rank_id": "6"
            },
            {
               "device_id": "7",
               "device_ip": "192.168.1.17",
               "rank_id": "7"
            },
            {
               "device_id": "8",
               "device_ip": "192.168.1.18",
               "rank_id": "8"
            },
            {
               "device_id": "9",
               "device_ip": "192.168.1.19",
               "rank_id": "9"
            },
            {
               "device_id": "10",
               "device_ip": "192.168.1.20",
               "rank_id": "10"
            },
            {
               "device_id": "11",
               "device_ip": "192.168.1.21",
               "rank_id": "11"
            },
            {
               "device_id": "12",
               "device_ip": "192.168.1.22",
               "rank_id": "12"
            },
            {
               "device_id": "13",
               "device_ip": "192.168.1.23",
               "rank_id": "13"
            },
            {
               "device_id": "14",
               "device_ip": "192.168.1.24",
               "rank_id": "14"
            },
            {
               "device_id": "15",
               "device_ip": "192.168.1.25",
               "rank_id": "15"
            }
         ],
         "server_id": "10.0.0.1",
         "container_ip": "10.0.0.1"
      },
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "192.168.2.10",
               "rank_id": "16"
            },
            {
               "device_id": "1",
               "device_ip": "192.168.2.11",
               "rank_id": "17"
            },
            {
               "device_id": "2",
               "device_ip": "192.168.2.12",
               "rank_id": "18"
            },
            {
               "device_id": "3",
               "device_ip": "192.168.2.13",
               "rank_id": "19"
            },
            {
               "device_id": "4",
               "device_ip": "192.168.2.14",
               "rank_id": "20"
            },
            {
               "device_id": "5",
               "device_ip": "192.168.2.15",
               "rank_id": "21"
            },
            {
               "device_id": "6",
               "device_ip": "192.168.2.16",
               "rank_id": "22"
            },
            {
               "device_id": "7",
               "device_ip": "192.168.2.17",
               "rank_id": "23"
            },
            {
               "device_id": "8",
               "device_ip": "192.168.2.18",
               "rank_id": "24"
            },
            {
               "device_id": "9",
               "device_ip": "192.168.2.19",
               "rank_id": "25"
            },
            {
               "device_id": "10",
               "device_ip": "192.168.2.20",
               "rank_id": "26"
            },
            {
               "device_id": "11",
               "device_ip": "192.168.2.21",
               "rank_id": "27"
            },
            {
               "device_id": "12",
               "device_ip": "192.168.2.22",
               "rank_id": "28"
            },
            {
               "device_id": "13",
               "device_ip": "192.168.2.23",
               "rank_id": "29"
            },
            {
               "device_id": "14",
               "device_ip": "192.168.2.24",
               "rank_id": "30"
            },
            {
               "device_id": "15",
               "device_ip": "192.168.2.25",
               "rank_id": "31"
            }
         ],
         "server_id": "10.0.0.2",
         "container_ip": "10.0.0.2"
      }
   ],
   "status": "completed",
   "version": "1.0"
}
```

---

### 步骤四：修改文件权限

`rank_table_file.json` 配置完成后，需要修改文件权限为 640（即仅文件所有者可读写，同组用户只读，其他用户无权限），以保障文件安全性：

```shell
chmod 640 /path/to/rank_table_file.json
```

---

### 步骤五：在服务化部署中使用

配置完 `rank_table_file.json` 后，在启动多机推理服务时，需要通过环境变量指定该文件的路径。在每台机器上设置以下环境变量：

```shell
export RANK_TABLE_FILE="/path/to/rank_table_file.json"
export MASTER_IP=xxx.xxx.xxx.xxx           # 主节点 IP（即 server_list 中第一个 server 的 server_id）
export MIES_CONTAINER_IP=xxx.xxx.xxx.xxx   # 本机 IP
export MASTER_PORT=xxxx                    # 主机端口号，取值范围 [0, 65535]，且不与本机其他服务端口冲突
```

> [!NOTE]
>
> - `MASTER_IP` 应填写 `server_list` 中第一个 server 的 `server_id` 值。
> - `MIES_CONTAINER_IP` 应填写当前机器对应的 `server_id` 值。
> - 如果容器内设置了代理，需要取消设置，避免多机通信异常：
>
>     ```shell
>     unset http_proxy
>     unset https_proxy
>     ```

更多服务化部署的环境变量配置，请参见[环境变量说明](./environment_variable.md)和[配置参数说明（服务化）](./service_parameter_configuration.md)。

---

## 常见问题

### 1. HCCL 连接超时

**现象**：服务拉起失败，日志中出现 HCCL 连接超时错误。

**解决方案**：增大 HCCL 连接超时时间：

```shell
export HCCL_CONNECT_TIMEOUT=7200
```

### 2. TLS 校验报错

**现象**：HCCL 初始化时报 TLS 相关错误。

**解决方案**：检查所有机器上每张卡的 TLS 校验行为是否一致，建议统一设置为 0：

```shell
for i in {0..15}; do hccn_tool -i $i -tls -s enable 0; done
```

### 3. rank_table_file.json 权限不足

**现象**：服务启动时报文件权限错误。

**解决方案**：确认文件权限已设置为 640：

```shell
chmod 640 /path/to/rank_table_file.json
```

### 4. 普通用户权限部署

如果以普通用户权限部署服务化（例如，用户名为 HwHiAiUser，用户 ID 为 1001），需要修改模型目录及目录下文件的属主为 1001（root 用户权限可忽略），并修改权重目录权限为 750：

```shell
chown -R 1001:1001 /path/to/weights
chmod 750 /path/to/weights
```
