# 共享专家混置

- 共享专家外置：共享专家独立部署在固定的前几张NPU卡上，与路由专家/冗余专家分离。计算负载均衡时只考虑路由专家。

    计算流程：dispatch -\> 同时计算共享专家和路由专家 -\> combine

- 共享专家内置：共享专家和路由专家/冗余专家部署在同一张NPU上。计算负载均衡时只考虑路由专家。

    计算流程：共享专家matmul -\> dispatch -\> 路由专家 -\> combine -\> 共享专家结果 + 路由专家结果

- 共享专家混置：把共享专家当做路由专家来计算负载均衡。

    计算流程：dispatch -\> 同时计算共享专家和路由专家 -\> combine

## 限制与约束

- 仅支持DeepSeek V3/R1。
- 仅Atlas 800I A3 超节点服务器的144卡场景，支持单独设置共享专家外置。如果该场景搭配负载均衡使用，则性能更优。
- 共享专家混置可单独设置，如果该场景搭配负载均衡使用，则性能更优。
- 共享专家外置只支持Atlas 800I A3 超节点服务器；共享专家混置同时支持Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器。

## 使用样例

- （推荐）搭配专家负载均衡
    1.  请参见[冗余专家部署表生成](./expert_parallelism_load_balancer.md#冗余专家部署表生成)，生成专家部署表。
    2.  在配置文件中修改如下参数。

        ```json
                "models": {
                  "deepseekv2": {
                    "ep_level": 2,
                    "eplb": {
                      "level": 1,
                      "expert_map_file": "xxxx.json"
                    }
                  }
                }
        ```

- Atlas 800I A3 超节点服务器  144卡单独使用共享专家外置，且不搭配专家负载均衡。

    在配置文件中修改如下参数。

    ```json
            "models": {
              "deepseekv2": {
                "ep_level": 2,
                "num_dangling_shared_experts": 32
              }
             }
    ```

- 单独设置共享专家混置：

    在配置文件中修改如下参数。

    ```json
            "models": {
              "deepseekv2": {
                "mix_shared_routing": true
              }
             }
    ```

## 执行推理

1. 配置服务化参数。服务化的config.json文件路径的详细说明请参考[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)。具体参数配置请参见[使用样例](#使用样例)。
2. 启动服务。具体请参考《MindIE Motor开发指南》中的“快速入门 \> [启动服务](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/quick_start.md)”章节。

