# rank_table_file.json Configuration Guide

## Overview

In multi-node inference, MindIE-LLM relies on Huawei Collective Communication Library (HCCL) for collective communication across nodes. To enable NPU cards on different nodes to discover each other and establish communication, a `rank_table_file.json` must be provided. This file describes the topology information of each card in the cluster.

`rank_table_file.json` is essentially a "communication topology table" that provides HCCL with the following key information:

- Number of machines in the cluster (`server_count`)

- NPU device IDs per machine (`device_id`)

- Network IP address of each NPU device (`device_ip`)

- Global index of each NPU device (`rank_id`)

- Network IP address of each machine (`server_id`)

This document describes how to perform network checks, obtain IP addresses, write the `rank_table_file.json` file, and configure related permissions.

## Prerequisites

- MindIE-LLM has been installed. For details, see [Installation Guide](../install/installing_MindIE.md).

- The NPU status of each node in the cluster has been confirmed as normal. You can check it using the `npu-smi info` command.

- In a multi-node scenario, the network between nodes is interconnected.

## Procedure

### Step 1: Checking the Machine Network Status

Before configuring `rank_table_file.json`, it is recommended to check the NPU network status of each node to ensure that the physical links are connected, the network is healthy, and the gateway and TLS configurations are correct. The following commands need to be executed on each machine (using an A3 environment with 16 NPUs as an example).

> [!NOTE]
>
> The following commands all use `hccn_tool`, which is provided with the CANN installation package. If the command is unavailable, check whether the CANN environment is correctly configured.

1. Check physical link connectivity.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done
    ```

2. Check link status.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -link -g; done
    ```

3. Check network health.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -net_health -g; done
    ```

4. Check whether the detection IP configuration is correct.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -netdetect -g; done
    ```

5. Check whether the gateway configuration is correct.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -gateway -g; done
    ```

6. Check NPU TLS verification behavior for consistency.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -tls -g; done | grep switch
    ```

7. Unify NPU TLS verification behavior to `0`.

    ```shell
    for i in {0..15}; do hccn_tool -i $i -tls -s enable 0; done
    ```

> [!NOTE]
>
> It is recommended to uniformly set the TLS verification behavior of all cards to 0 to avoid HCCL errors.

### Step 2: Obtaining the IP Address of Each NPU

Run the following command on each machine to obtain the IP address corresponding to each NPU on that machine:

```shell
for i in {0..15}; do hccn_tool -i $i -ip -g; done
```

After execution, the IP address of each card will be printed. Record the output for later use when writing the `rank_table_file.json`.

> [!NOTE]
>
> - Each NPU card has a dedicated RDMA over Converged Ethernet (RoCE) interface with its own IP address, used for high-speed inter-card communication.  
> - The number of NPUs may vary across machines—adjust the loop range accordingly. For example, use `{0..7}` for 8 cards per machine.

### Step 3: Writing `rank_table_file.json`

#### File Format Description

`rank_table_file.json` is a JSON-formatted configuration file that contains the topology information for all nodes and all NPUs in the cluster.

**Table 1** `rank_table_file.json` parameter description

| Parameter | Description |
|------|------|
| server_count | Total number of nodes (i.e., number of machines). |
| server_list | List of nodes, where the first server is the Master node. |
| device_id | ID of the current card within the local machine, with a value range of [0, number of local cards). |
| device_ip | IP address of the current card, obtained using the `hccn_tool` command. |
| rank_id | Unique ID of the current card across all cards in all machines, with a value range of [0, total number of cards). |
| server_id | IP address of the current node (i.e., the machine's network IP). |
| container_ip | Container IP address (required for serving deployment). If there is no special configuration, it is the same as `server_id`. |
| status | Fixed value: `"completed"`. |
| version | Fixed value: `"1.0"`. |

#### rank_id Numbering Rules

`rank_id` is the unique ID of each NPU globally, incrementing from 0. It is assigned in order of node and  card numbers.

For example, in a scenario with two machines each having 16 cards:

- Machine 1: rank IDs 0–15 for NPUs 0–15

- Machine 2: rank IDs 16–31 for NPUs 16–31

#### Dual-Machine Example

The following is a complete example for two machines, each with 16 NPUs. Users need to replace the `device_ip`, `server_id`, and `container_ip` fields with the actual IP addresses.

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

### Step 4: Modifying File Permissions

After the `rank_table_file.json` configuration is complete, you need to modify the file permissions to `640` (i.e., only the file owner can read and write, users in the same group can read only, and other users have no permissions) to ensure file security:

```shell
chmod 640 /path/to/rank_table_file.json
```

### Step 5: Using for Serving Deployment

After configuring `rank_table_file.json`, you need to specify the path to this file through environment variables when starting the multi-machine inference service. Set the following environment variables on each machine:

```shell
export RANK_TABLE_FILE="/path/to/rank_table_file.json"
export MASTER_IP=xxx.xxx.xxx.xxx           # IP address of the primary node (i.e., the `server_id` of the first server in `server_list`)
export MIES_CONTAINER_IP=xxx.xxx.xxx.xxx   # IP address of the local machine
export MASTER_PORT=xxxx                    # Host port number, ranging from [0, 65535], and must not conflict with other service ports on the local machine
```

> [!NOTE]
>
> - `MASTER_IP` should be set to the `server_id` value of the first server in `server_list`.
> - `MIES_CONTAINER_IP` should be set to the `server_id` value corresponding to the current machine.
> - If a proxy is configured within the container, it needs to be unset to prevent multi-machine communication anomalies:
>
>     ```shell
>     unset http_proxy
>     unset https_proxy
>     ```

For more environment variable configurations for serving deployment, see [Environment Variable Description](./environment_variable.md) and [Service Parameter Configuration Description](./service_parameter_configuration.md).

## FAQs

### 1. HCCL Connection Timeout

**Symptom**: The service fails to start, and an HCCL connection timeout error appears in the logs.

**Solution**: Increase the HCCL connection timeout period.

```shell
export HCCL_CONNECT_TIMEOUT=7200
```

### 2. TLS Validation Error

**Symptom**: A TLS-related error is reported during HCCL initialization.

**Solution**: Check whether the TLS validation behavior of each card on all machines is consistent. It is recommended to set it to 0 uniformly.

```shell
for i in {0..15}; do hccn_tool -i $i -tls -s enable 0; done
```

### 3. Insufficient Permissions for `rank_table_file.json`

**Symptom**: A file permission error is reported when the service starts.

**Solution**: Verify that the file permissions are set to `640`.

```shell
chmod 640 /path/to/rank_table_file.json
```

### 4. Deployment with Non-Root User Permissions

If deploying the service with non-root user permissions (for example, username `HwHiAiUser` with user ID `1001`), you need to change the owner of the model directory and its files to `1001` (root user permissions can be ignored), and change the permissions of the weights directory to `750`:

```shell
chown -R 1001:1001 /path/to/weights
chmod 750 /path/to/weights
```
