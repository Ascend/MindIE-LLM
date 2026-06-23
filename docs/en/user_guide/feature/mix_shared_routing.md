# Mixed Deployment of Shared Experts

- External shared expert: Shared experts are independently deployed on the first several NPUs and are separated from routing and redundant experts. During load balancing, only routing experts are included in the computation.

    Computation process: dispatch -\> simultaneous computing of shared and routing experts -\> combine

- Built-in shared experts: Shared experts and routing/redundant experts are deployed on a single NPU. During load balancing, only routing experts are included in the computation.

    Computation process: matmul for shared experts -\> dispatch -\> routing experts -\> combine -\> results of shared and routing experts

- Mixed deployment of shared experts: Shared experts are used as routing experts for load balancing.

    Computation process: dispatch -\> simultaneous computing of shared and routing experts -\> combine

## Constraints

- Only DeepSeek V3/R1 is supported.
- External shared experts can be configured independently only for the 144-card Atlas 800I A3 SuperPoD servers. Performance improves if load balancing is enabled in this scenario.
- The mixed deployment of shared experts can be set separately. Performance improves if load balancing is enabled in this scenario.
- External shared experts are supported only by Atlas 800I A3 SuperPoD servers. Mixed deployment of shared experts is supported by both Atlas 800I A2 inference servers and Atlas 800I A3 SuperPoD servers.

## Usage Examples

- (Recommended) Enable expert load balancing.
    1. Generate the expert deployment table. For details, see [Generating a Redundant Expert Deployment Table](./expert_parallelism_load_balancer.md#generate-a-deployment-table-of-redundant-experts).
    2. Modify the following parameters in the configuration file.

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

- The 144-card Atlas 800I A3 SuperPoD servers use external shared experts independently with expert load balancing disabled.

    Modify the following parameters in the configuration file.

    ```json
            "models": {
              "deepseekv2": {
                "ep_level": 2,
                "num_dangling_shared_experts": 32
              }
             }
    ```

- Independent configuration of mixed deployment of shared experts:

    Modify the following parameters in the configuration file.

    ```json
            "models": {
              "deepseekv2": {
                "mix_shared_routing": true
              }
             }
    ```

## Inference

1. Set serving parameters. For details about the path of the `config.json` file, see [Configuration Parameters (Serving)](../user_manual/service_parameter_configuration.md). For details about parameter settings, see [Examples](#usage-examples).
2. Start the service. For details, see "Quick Start" \> "[Starting the Service](https://gitcode.com/Ascend/MindIE-Motor/blob/v3.0.0/docs/zh/user_guide/quick_start.md)" in *MindIE Motor Developer Guide*.
