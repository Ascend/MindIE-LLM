# PD Co-location

## PD Co-location on a Single Node

### Prerequisites

- The NPU driver and firmware, CANN, PyTorch, ATB Models, and MindIE have been installed on the server or in the container.
- If HTTPS two-way authentication is enabled, prepare the service certificate, server private key, and signature verification certificate in advance.
- If you start the inference service in containerized mode, ensure that the shared memory must be greater than or equal to 1 GB.
- The Server requires Python 3.11.x.

### Procedure (using the `.whl` package)

1. Go to the MindIE installation directory as the installation user.

    ```bash
     cd {MindIE installation directory}
     ```

2. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.

    ```bash
    chmod 640 mindie_llm/conf/config.json
    ```

    > [!NOTE]
    > If the file permission does not meet the requirements, the Server will fail to be started.

3. Set parameters as required.

   Before the configuration, pay attention to the following points:

   | Name             | Description                                            | Precautions                                                    |
   | --------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
   | httpsEnabled          | Enable HTTPS communication (that is, when `httpsEnabled` is set to `true`).       | If this function is disabled, high network security risks exist.                            |
   | maxLinkNum            | The default value is `1000`. You are advised to set it to `300`.                     | This parameter is affected by model performance. Typically, 1,000 concurrent requests can be used for a small model with short sequence lengths.|
   | MIES_CONFIG_JSON_PATH | You can set this environment variable to provide the configuration file of the Server.  | You need to ensure the security of the configuration file.                          |
   | modelWeightPath       | Model weight path. All files in this path are provided by users.  | You need to ensure the security of all files in this path. In addition, the `config.json` file in this path must have its user group and username match the current user, be a regular file (not a symlink), and have permissions no more permissive than `750`. Failure to meet these requirements will cause the Server to fail to start.|
   | tlsCaFile             | Service certificate file used by the RESTful interface of the service plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | tlsCert               | Service certificate file used by the RESTful interface of the service plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | tlsPk                 | Service certificate private key file used by the RESTful interface of the service plane.         | You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | tlsCrlFiles           | CRL files used by the RESTful interface of the service plane.         | The files are provided by users. Users need to ensure the security of these files.      |
   | managementTlsCaFile   | List of CA certificate files used by the RESTful interface of the management plane.           | The files are provided by users. Users need to ensure the security of these files.      |
   | managementTlsCert     | Service certificate file used by the RESTful interface of the management plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | managementTlsPk       | Private key file of the service certificate used by the RESTful interface of the management plane.         | You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | managementTlsCrlFiles | CRL files used by the RESTful interface of the management plane.         | The files are provided by users. Users need to ensure the security of these files.      |
   | interCommTlsCaFiles   | List of CA certificate files used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.  | The files are provided by users. Users need to ensure the security of these files.      |
   | interCommTlsCert      | Service certificate file used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.    | The file is provided by yourself. You need to ensure the security of the file.              |
   | interCommPk           | Private key file of the service certificate used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.| You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | interCommTlsCrlFiles  | CRL files used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.| The files are provided by users. Users need to ensure the security of these files.      |
   | interNodeTlsCaFiles   | CA certificate files used for communication between the primary and secondary nodes in the multi-node scenario.  | The files are provided by users. Users need to ensure the security of these files.      |
   | interNodeTlsCert      | Service certificate file used for communication between the primary and secondary nodes in the multi-node cluster scenario    | The file is provided by yourself. You need to ensure the security of the file.              |
   | interNodeTlsPk        | Private key file of the service certificate used for communication between the primary and secondary nodes in the multi-node scenario.| You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | interNodeTlsCrlFiles  | CRL files used for communication between the primary and secondary nodes in the multi-node scenario.| The files are provided by users. Users need to ensure the security of these files.      |

   a. Go to the `conf` directory and open the `config.json` file.

    ```bash
    cd mindie_llm/conf
    vim config.json
    ```

   b. Press **i** to enter the insert mode and modify parameters as required. For details about the parameters, see [Configuration Parameters (Serving)](service_parameter_configuration.md).

   c. Press **Esc**, type **:wq!**, and press **Enter** to save the settings and exit.

4. (Optional) Enable HTTPS authentication (that is, set `httpsEnabled` to `true`).

   a. Import certificates. [Table 1](#table1) describes the certificate information.

    > [!NOTE]
    > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
    > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
    > - When importing certificates, ensure that the script permissions required for CA certificates, service certificates, private key certificates, and CRL certificates are 600, 400, and 600, respectively.
    > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

      Table 1 Certificate files <a id="table1"></a>

      | Certificate File              | Default Path                                            | Description                                             |
      | ---------------------- | -------------------------------------------------------| ------------------------------------------------- |
      | Root certificate                | {MindIE installation directory}/latest/mindie-service/security/ca/   | Multiple CA certificates are supported.<br><br>Required when HTTPS is enabled.      |
      | Service certificate              | {MindIE installation directory}/latest/mindie-service/security/certs/| Required when HTTPS is enabled.                                |
      | Private key of a service certificate           | {MindIE installation directory}/latest/mindie-service/security/keys/ | Private key file encryption is supported.<br><br>Required when HTTPS is enabled.|
      | Service CRL       | {MindIE installation directory}/latest/mindie-service/security/certs/ | Optional when HTTPS is enabled.                                |

   b. Run the following commands in *{MindIE installation directory}* to modify the user permissions on the certificate files:

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
      ```

5. Configure environment variables.

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh                                 # CANN
    source /usr/local/Ascend/nnal/atb/set_env.sh                                       # ATB
    source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh               # ATB Models
    ```

6. Copy the model weight file (prepared by yourself) to the directory specified by `modelWeightPath` in `config.json`.

    ```shell
    cp -r {Path_to_the_model_weight_file} {modelWeightPath}
    ```

7. Configure the environment variables for running the ATB Models.

    ```shell
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

8. Start the service.

    > [!NOTE]
    > Before starting the service, you are advised to use the pre-check tool of MindStudio to verify the fields in the configuration file and check the validity of the configuration. For details, see [msprechecker](https://gitcode.com/Ascend/msit/tree/master/msprechecker).

    Start the service directly.

    ```shell
    mindie_llm_server
    ```

     If the following information is displayed, the service is started successfully.

    ```text
    Daemon start success!
    ```

 > [!NOTE]
 >
 > - In the directory where the service is started, Ascend-CANN-Toolkit generates a `kernel_meta_temp_*xxxx*` directory to store the CCE files of operators. Therefore, start the inference service in a directory where the current user has write permissions, such as the `Ascend-mindie-server_{version}_linux-{arch}_{abi}` directory, or a temporary directory manually created under `Ascend-mindie-server_{version}_linux-{arch}`.
 > - To switch to another user, run the `rm -f /dev/shm/*` command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
 > - The `output.log` file captured by the standard output stream supports user-defined files and paths.
 > - If service startup fails due to missing `lib*.so` dependencies, refer to "`libboost_thread.so.1.82.0` Cannot Be Found When MindIE Motor Is Started".
 > - You are not advised to repeatedly start the service in the same container. Before repeatedly starting the service, delete the `*llm_backend_*` and `llm_tokenizer_shared_memory_*` files in the `/dev/shm/` directory of the container. The following commands are used as an example:
 >
>      ```shell
>      find /dev/shm -name '*llm_backend_*' -type f -delete
>      find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
>      ```

### Procedure (using the `.run` package)

1. Go to the MindIE installation directory as the installation user.

    ```bash
     cd {MindIE installation directory}
     ```

2. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.

    ```bash
   chmod 750 mindie-service
   chmod -R 550 mindie-service/bin
   chmod -R 500 mindie-service/bin/mindie_llm_backend_connector
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

    > [!NOTE]
    > If the file permission does not meet the requirements, the Server will fail to be started.

3. Set parameters as required.

   Before the configuration, pay attention to the following points:

   | Name             | Description                                            | Precautions                                                    |
   | --------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
   | httpsEnabled          | Enable HTTPS communication (that is, when `httpsEnabled` is set to `true`).       | If this function is disabled, high network security risks exist.                            |
   | maxLinkNum            | The default value is `1000`. You are advised to set it to `300`.                     | This parameter is affected by model performance. Typically, 1,000 concurrent requests can be used for a small model with short sequence lengths.|
   | MIES_CONFIG_JSON_PATH | You can set this environment variable to provide the configuration file of the Server.  | You need to ensure the security of the configuration file.                          |
   | modelWeightPath       | Model weight path. All files in this path are provided by users.  | You need to ensure the security of all files in this path. In addition, the `config.json` file in this path must have its user group and username match the current user, be a regular file (not a symlink), and have permissions no more permissive than `750`. Failure to meet these requirements will cause the Server to fail to start.|
   | tlsCaFile             | Service certificate file used by the RESTful interface of the service plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | tlsCert               | Service certificate file used by the RESTful interface of the service plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | tlsPk                 | Service certificate private key file used by the RESTful interface of the service plane.         | You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | tlsCrlFiles           | CRL files used by the RESTful interface of the service plane.         | The files are provided by users. Users need to ensure the security of these files.      |
   | managementTlsCaFile   | List of CA certificate files used by the RESTful interface of the management plane.           | The files are provided by users. Users need to ensure the security of these files.      |
   | managementTlsCert     | Service certificate file used by the RESTful interface of the management plane.             | The file is provided by yourself. You need to ensure the security of the file.              |
   | managementTlsPk       | Private key file of the service certificate used by the RESTful interface of the management plane.         | You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | managementTlsCrlFiles | CRL files used by the RESTful interface of the management plane.         | The files are provided by users. Users need to ensure the security of these files.      |
   | interCommTlsCaFiles   | List of CA certificate files used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.  | The files are provided by users. Users need to ensure the security of these files.      |
   | interCommTlsCert      | Service certificate file used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.    | The file is provided by yourself. You need to ensure the security of the file.              |
   | interCommPk           | Private key file of the service certificate used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.| You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | interCommTlsCrlFiles  | CRL files used for communication between the prefill and decoding nodes in the prefill-decoding disaggregation scenario.| The files are provided by users. Users need to ensure the security of these files.      |
   | interNodeTlsCaFiles   | CA certificate files used for communication between the primary and secondary nodes in the multi-node scenario.  | The files are provided by users. Users need to ensure the security of these files.      |
   | interNodeTlsCert      | Service certificate file used for communication between the primary and secondary nodes in the multi-node cluster scenario    | The file is provided by yourself. You need to ensure the security of the file.              |
   | interNodeTlsPk        | Private key file of the service certificate used for communication between the primary and secondary nodes in the multi-node scenario.| You are advised to use the encrypted private key file. The file is provided by yourself. You need to ensure the security of the file.|
   | interNodeTlsCrlFiles  | CRL files used for communication between the primary and secondary nodes in the multi-node scenario.| The files are provided by users. Users need to ensure the security of these files.      |

   a. Go to the `conf` directory and open the `config.json` file.

    ```bash
    cd mindie-service/conf
    vim config.json
    ```

   b. Press **i** to enter the insert mode and modify parameters as required. For details about the parameters, see [Configuration Parameters (Serving)](service_parameter_configuration.md).

   c. Press **Esc**, type **:wq!**, and press **Enter** to save the settings and exit.

4. (Optional) Enable HTTPS authentication (that is, set `httpsEnabled` to `true`).

   a. Import certificates. [Table 2](#table2) describes the certificate information.

    > [!NOTE]
    > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
    > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
    > - When importing certificates, ensure that the script permissions required for CA certificates, service certificates, private key certificates, and CRL certificates are 600, 400, and 600, respectively.
    > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

      Table 2 Certificate files <a id="table2"></a>

      | Certificate File              | Default Path                                            | Description                                             |
      | ---------------------- | -------------------------------------------------------| ------------------------------------------------- |
      | Root certificate                | {MindIE installation directory}/latest/mindie-service/security/ca/   | Multiple CA certificates are supported.<br><br>Required when HTTPS is enabled.      |
      | Service certificate              | {MindIE installation directory}/latest/mindie-service/security/certs/| Required when HTTPS is enabled.                                |
      | Private key of a service certificate           | {MindIE installation directory}/latest/mindie-service/security/keys/ | Private key file encryption is supported.<br><br>Required when HTTPS is enabled.|
      | Service CRL       | {MindIE installation directory}/latest/mindie-service/security/certs/ | Optional when HTTPS is enabled.                                |

   b. Run the following commands in *{MindIE installation directory}* to modify the user permissions on the certificate files:

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
      ```

5. Configure environment variables.

    ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh                                 # CANN
   source /usr/local/Ascend/nnal/atb/set_env.sh                                       # ATB
   source /usr/local/Ascend/atb-models/set_env.sh                                     # ATB Models
    ```

6. Copy the model weight file (prepared by yourself) to the directory specified by `modelWeightPath` in `config.json`.

    ``` shell
    cp -r {Path_to_the_model_weight_file} {modelWeightPath}
    ```

7. Go to the ```{MindIE_installation_directory}/latest``` directory and load environment variables.

    ```bash
   cd ../../
   source mindie-service/set_env.sh
    ```

8. Start the service. The startup command must be run in the ```/{MindIE installation directory}/latest/mindie-service``` directory.

    > [!NOTE]
    > Before starting the service, you are advised to use the pre-check tool of MindStudio to verify the fields in the configuration file and check the validity of the configuration. For details, see [msprechecker](https://gitcode.com/Ascend/msit/tree/master/msprechecker).

   - (Recommended) Start the service in background process mode.
  
   ```bash
   nohup ./bin/mindieservice_daemon > output.log 2>&1 &
   ```

   If the following information is printed in the file captured by the standard output stream, the startup is successful:

   ```text
   Daemon start success!
   ```

   - Start the service directly. 

   ```bash
   ./bin/mindieservice_daemon
   ```

   If the following information is displayed, the service is started successfully.

   ```text
   Daemon start success!
   ```

 > [!NOTE]
 >
 > - Ascend-CANN-Toolkit generates the kernel_meta_temp_xxxx directory in the directory where the service is started. This directory stores the CCE file of the operator. Therefore, you need to start the inference service in the directory on which the current user has the write permission (for example, Ascend-mindie-server_{version}_linux-{arch}_{abi} or a temporary directory in Ascend-mindie-server_{version}_linux-{arch})..
 > - To switch to another user, run the rm -f /dev/shm/* command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
 > - For security, the permission on the bin directory is 550, and the directory does not have the write permission. Therefore, mindieservice_daemon cannot be started in the bin directory.
 > - The output.log file captured by the standard output stream supports user-defined files and paths.
 > - You are not advised to repeatedly start the service in the same container. Before repeatedly starting the service, delete the `*llm_backend_*` and `llm_tokenizer_shared_memory_*` files in the `/dev/shm/` directory of the container. The following commands are used as an example:
 >
>      ```shell
>      find /dev/shm -name '*llm_backend_*' -type f -delete
>      find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
>      ```

## PD Co-location Across Multiple Nodes

If the weight of a single model is too large and the memory of a single inference server is limited, the weight parameters of the entire model cannot be accommodated. In this case, multi-node inference is required.

### Prerequisites

- The Server requires Python 3.11.x. Python 3.11 is used as an example in this section. If Python 3.11 is not the default version, add the following environment variables (use the actual Python path):

    ```linux
    export LD_LIBRARY_PATH=/usr/local/python3.11/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/python3.11/bin:$PATH
    ```

- The NPU driver and firmware, CANN, PyTorch, ATB Models, and MindIE have been installed on the server or container.

- If you start the inference service in containerized mode, ensure that the shared memory must be greater than or equal to 1 GB.

- If two-way HTTPS authentication or multi-node communication authentication is enabled, prepare the service certificate, server private key, signature verification certificate, etc. in advance. For details, see "Cluster Service Deployment" > "Single-Node (Non-Distributed) Service Deployment" > "Installation and Deployment" > "Example of Deploying Services Using Deployer" > "Deploying the Deployer Server" > "[Preparing the TLS Certificate](https://gitcode.com/Ascend/MindIE-Motor-CPP/blob/v3.0.0/docs/en/user_guide/service_deployment/single_machine_service_deployment.md)" in *MindIE Motor Development Guide*.

### Constraints

- Only the Atlas 800I A2 inference server environment is supported. A maximum of four servers and 32 cards are supported. For details about the models supported by multi-server inference, see [Model List](../model_support_list.md). The Atlas 300I Duo inference card environment is not supported.
- The default value of `maxLinkNum` is `1000`. You are advised to set it to `300`. This parameter is affected by model performance. Typically, 1,000 concurrent requests can be used for a small model with short sequence lengths.
- The default sampling parameters for the weights of different nodes must be consistent. If the sampling parameters are not configured, the inference service may be suspended.

### Related Environment Variables

| Name             | Description                                                    |
| --------------------- | ------------------------------------------------------------ |
| MIES_CONTAINER_IP     | For containerized deployment, set this parameter to the IP address of the container. If the container shares an IP address with a bare metal server, set this parameter to the IP address of the bare metal server. This IP address is used for Google Remote Procedure Call (gRPC) between multiple nodes and for request receiving on the EndPoint's service plane. This parameter is not required for bare metal deployment.|
| HOST_IP               | For bare metal deployment (not recommended), set this parameter to the IP address of the PM or VM. This parameter is not configured for containerized deployment.|
| RANK_TABLE_FILE       | Absolute path to the ranktable.json file.  Mandatory for multi-node inference. You are advised to run the `unset RANK_TABLE_FILE` command to cancel this environment variable for single-node inference. If it needs to be set, the file content must be correct and valid (the node IP address and device IP must be correct). Otherwise, the model initialization will fail.|
| MIES_CONFIG_JSON_PATH | Path to the `config.json` file. If the environment variable exists, its value is read. If the environment variable does not exist, the `${MINDIE_LLM_HOME_PATH}/conf/config.json` file is read.|
| HCCL_DETERMINISTIC    | Deterministic computation of HCCL communication. For multi-node inference, you are advised to set this parameter to `true`.          |

> [!NOTE]
> When the Server is started, the system determines whether to perform single-node or multi-node inference based on the value of `multiNodesInferEnabled`.
>
> - `multiNodesInferEnabled` = `false`: single-node inference. The Server does not read the `RANK_TABLE_FILE` environment variable during startup. However, when the underlying model acceleration library is initialized, it attempts to read this environment variable. Therefore, in the single-node inference scenario, if this environment variable is set, ensure that the file content is correct (that is, server_count=1; node IP address, device_ip, and rank_id must be correct).
> - `multiNodesInferEnabled` = `true`: multi-node inference.
>   - During the server startup, the `RANK_TABLE_FILE` environment variable is read and the system checks whether the content of the ranktable file is valid.
>   - When multi-node inference is enabled, `npuDeviceIds` and `worldSize` in the `config.json` file become invalid. The card IDs in use and the total number of ranks are determined based on the ranktable file.
> - The node whose rank ID is 0 is the Master node, and the other nodes are Slave nodes.
> - The Master service instance can receive inference requests from users, while the Slave service instance cannot.

### Example of the Ranktable File

The permission on the `ranktable.json` file must be set to `640`. For details, see the following example. (This file needs to be compiled by yourself.)

   ```json
   {
      "version": "1.0",
      "server_count": "2",
      "server_list": [
         {
               "server_id": "IP address of the Master node",
               "container_ip": "Container IP address of the Master node",
               "device": [
                  { "device_id": "0", "device_ip": "10.20.0.2", "rank_id": "0" },
                  { "device_id": "1", "device_ip": "10.20.0.3", "rank_id": "1" },
                  { "device_id": "2", "device_ip": "10.20.0.4", "rank_id": "2" },
                  { "device_id": "3", "device_ip": "10.20.0.5", "rank_id": "3" },
                  { "device_id": "4", "device_ip": "10.20.0.6", "rank_id": "4" },
                  { "device_id": "5", "device_ip": "10.20.0.7", "rank_id": "5" },
                  { "device_id": "6", "device_ip": "10.20.0.8", "rank_id": "6" },
                  { "device_id": "7", "device_ip": "10.20.0.9", "rank_id": "7" }
               ]
         },
         {
               "server_id": "IP address of the Slave node",
               "container_ip": "IP address of the container on the Slave node",
               "device": [
                  { "device_id": "0", "device_ip": "10.20.0.10", "rank_id": "8" },
                  { "device_id": "1", "device_ip": "10.20.0.11", "rank_id": "9" },
                  { "device_id": "2", "device_ip": "10.20.0.12", "rank_id": "10" },
                  { "device_id": "3", "device_ip": "10.20.0.13", "rank_id": "11" },
                  { "device_id": "4", "device_ip": "10.20.0.14", "rank_id": "12" },
                  { "device_id": "5", "device_ip": "10.20.0.15", "rank_id": "13" },
                  { "device_id": "6", "device_ip": "10.20.0.16", "rank_id": "14" },
                  { "device_id": "7", "device_ip": "10.20.0.17", "rank_id": "15" }
               ]
         }
      ],
      "status": "completed"
   }
   ```

Parameter description:

- IP address of the Master/Slave node: Change it based on the actual situation.
- Container IP address of the Master/Slave node: Generally, the IP address is the same as that of the master/slave node. If `--net=host` is used upon container startup, the IP address must be the same as the IP address of the host. Change the IP address as required.
- `device_id`: sequence number of the NPU on the actual node.
- `device_ip`: IP address of the NPU, which can be configured using hccn_tool.
- `rank_id`: rank ID of the inference process.

> [!NOTE]
> The `ranktable.json` file is configured via the `RANK_TABLE_FILE` environment variable. If users provide this file themselves, they are responsible for ensuring its security. The file must be created on both the Master and Slave nodes.

### Procedure (using the `.whl` package)

> [!NOTE]
> Perform the following operations on both the Master and Slave nodes.

1. Create and start a Docker container. The following uses the 8-card Ascend environment as an example.

   The following startup commands are for reference only. You can modify commands as required.

    ```bash
       docker run -it -d --net=host --shm-size=1g \
       --name container_name \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device=/dev/devmm_svm \
       --device=/dev/davinci0 \
       --device=/dev/davinci1 \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci4 \
       --device=/dev/davinci5 \
       --device=/dev/davinci6 \
       --device=/dev/davinci7 \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       -v /path-to-weights:/path-to-weights:ro \
       mindie:3.0.0-800I-A2-aarch64
   ```

2. Go to the MindIE installation directory as the installation user.

    ```bash
    cd {MindIE installation directory}
    ```

3. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.

    ```bash
    chmod 640 mindie_llm/conf/config.json
    ```

    > [!NOTE]
    > If the file permission does not meet the requirements, the Server will fail to be started.

4. Set parameters in the container as required.

   Before the configuration, see the note in Step 3.

   a. Go to the `conf` directory and open the `config.json` file.

      ```bash
      cd mindie_llm/conf
      vim config.json
      ```

   b. Press `i` to enter edit mode, set `"multiNodesInferEnabled"=true` to enable multi-node inference, and modify the parameters in Table 3 as needed. For details, see [Configuration Parameters (Serving)](service_parameter_configuration.md)..

      Table 3 Multi-node inference configuration

   | Configuration Item                | Configuration Description                                                    |
   | ---------------------- | ------------------------------------------------------------ |
   | multiNodesInferPort    | Port number for cross-node communication.                                          |
   | interNodeTLSEnabled    | Whether to enable certificate security authentication for cross-node communication. `true`: enabled. `false`: disabled. In this case, ignore the following parameters.|
   | interNodeTlsCaPath     | Path to the root certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCaFiles    | Root certificate name list. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCert       | Path to the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.        |
   | interNodeTlsPk         | Path to the private key file of the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.    |
   | interNodeTlsCrlPath    | Path to the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.|
   | interNodeTlsCrlFiles   | Name of the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.  |

   > [!NOTE]
   > - If HTTPS communication is disabled (·"httpsEnabled": false·), high network security risks exist.
   > - The `modelWeightPath` config file (`config.json`) must be owned by the current user (user/group match), not be a symlink, and have permissions no stricter than `640`. Otherwise, startup will fail.
   > - In a data center, if cross-node communication security authentication does not need to be enabled, set `interNodeTLSEnabled` to `false`. Disabling this option introduces significant network security risks.

   c. Press `Esc`, type `:wq!`, then press `Enter` to save and exit editing.

5. (Optional) If gRPC two-way authentication is enabled (that is, interNodeTLSEnabled is set to true),

    a. Import the certificate. [Table 4](#table4) describes the certificate files.

      > [!NOTE]
      > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
      > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
      > - When importing certificates, ensure that the permissions required by the CA certificate tool, service certificate tool, private key certificate tool, and CRL tool is 600, 600, 400, and 600, respectively.
      > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

    Table 4 Certificate file information <a id="table4"></a>

    | Certificate File              | Default Path                       | Description                                                        |
    | ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
    | Root certificate                | mindie-service/security/grpc/ca/    | Required when `interNodeTLSEnabled` is set to `true`.                    |
    | Service certificate              | mindie-service/grpc/certs/          | This parameter is mandatory when `interNodeTLSEnabled` is set to `true`.                    |
    | Private key of a service certificate          | mindie-service/security/grpc/keys/  | Private key file encryption is supported. Required when `interNodeTLSEnabled` is set to `true`.|
    | Service CRL      | mindie-service/security/grpc/certs/ | Required.                                                      |

    b. Run the following command in `{MindIE installation directory}/latest` to change the user permission on the certificate file:

    ```shell
    chmod 400 mindie-service/security/grpc/ca/*
    chmod 400 mindie-service/security/grpc/certs/*
    chmod 400 mindie-service/security/grpc/keys/*
    a. Import the certificate. For details about the certificate information, see [Table 5] (#table5).

    > [!NOTE]
    > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
    > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
    > - When importing certificates, ensure that the script permissions required for CA certificates, service certificates, private key certificates, and CRL certificates are 600, 400, and 600, respectively.
    > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

      Table 5 Certificate files <a id="table5"></a>

      | Certificate File              | Default Path                                            | Description                                             |
      | ---------------------- | -------------------------------------------------------- | ------------------------------------------------- |
      | Root certificate                | {MindIE installation directory}/latest/mindie-service/security/ca/   | Multiple CA certificates are supported.<br>Required when HTTPS is enabled.           |
      | Service certificate              | {MindIE installation directory}/latest/mindie-service/security/certs/| Required when HTTPS is enabled.                                |
      | Private key of a service certificate          | {MindIE installation directory}/latest/mindie-service/security/keys/ | Private key file encryption is supported.<br>Required when HTTPS is enabled.     |
      | Service CRL      | {MindIE installation directory}/latest/mindie-service/security/certs/  | Optional when HTTPS is enabled.                                |

    b. Run the following command in the `{MindIE installation directory}` to modify the user permission on the certificate files:

    ```bash
    chmod 400 mindie-service/security/ca/*
    chmod 400 mindie-service/security/certs/*
    chmod 400 mindie-service/security/keys/*
    ```

6. Configure environment variables.

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh                           # CANN
      source /usr/local/Ascend/nnal/atb/set_env.sh                                 # ATB
      source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh         # ATB Models
      ```

7. Copy the model weight file (prepared by yourself) to the directory specified by `modelWeightPath` in `config.json`.

      ```bash
      cp -r {Path_to_the_model_weight_file} {modelWeightPath}
      ```

8. Configure the environment variables for running the ATB Models.

    ```shell
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

9. Set the environment variables `RANK_TABLE_FILE` and `MIES_CONTAINER_IP` (e.g., using the rank table example from the [sample ranktable file](https://gitcode.com/Ascend/MindIE-Motor-CPP/blob/v3.0.0/docs/en/user_guide/service_deployment/pd_separation_service_deployment.md); see Table 4 for details).

    - Container corresponding to the master node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Master node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

    - Container corresponding to the Slave node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Slave node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

10. Start the service. This operation must be performed in containers on both the Master and Slave nodes.

    - Start the service directly.

      ```bash
      mindie_llm_server
      ```

      If the following information is displayed, the service is started successfully.

      ```text
      Daemon start success!
      ```

> [!NOTE]
>
> - In the directory where the service is started, Ascend-CANN-Toolkit generates a `kernel_meta_temp_*xxxx*` directory to store the CCE files of operators. Therefore, start the inference service in a directory where the current user has write permissions, such as the `Ascend-mindie-server_{version}_linux-{arch}_{abi}` directory, or a temporary directory manually created under `Ascend-mindie-server_{version}_linux-{arch}`.
> - To switch to another user, run the `rm -f /dev/shm/*` command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
> - The `output.log` file captured by the standard output stream supports user-defined files and paths.
> - If service startup fails due to missing `lib*.so` dependencies, refer to "`libboost_thread.so.1.82.0` Cannot Be Found When MindIE Motor Is Started".
> - You are not advised to repeatedly start the service in the same container. Before repeatedly starting the service, delete the `*llm_backend_*` and `llm_tokenizer_shared_memory_*` files in the `/dev/shm/` directory of the container. The following commands are used as an example:

   ```bash
   find /dev/shm -name '*llm_backend_*' -type f -delete
   find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
   ```

### Procedure (using the `.run` package)

> [!NOTE]
> Perform the following operations on both the Master and Slave nodes.

1. Create and start a Docker container. The following uses the 8-card Ascend environment as an example.

   The following startup commands are for reference only. You can modify commands as required.

    ```bash
       docker run -it -d --net=host --shm-size=1g \
       --name container_name \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device=/dev/devmm_svm \
       --device=/dev/davinci0 \
       --device=/dev/davinci1 \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci4 \
       --device=/dev/davinci5 \
       --device=/dev/davinci6 \
       --device=/dev/davinci7 \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       -v /path-to-weights:/path-to-weights:ro \
       mindie:3.0.0-800I-A2-aarch64
   ```

2. Go to the MindIE installation directory as the installation user.

    ```bash
    cd {MindIE installation directory}
    ```

3. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.

    ```bash
   chmod 750 mindie-service
   chmod -R 550 mindie-service/bin
   chmod -R 500 mindie-service/bin/mindie_llm_backend_connector
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

    > [!NOTE]
    > If the file permission does not meet the requirements, the Server will fail to be started.

4. Set parameters in the container as required.

   Before the configuration, see the note in Step 3.

   a. Go to the `conf` directory and open the `config.json` file.

      ```bash
      cd ../conf
      vim config.json
      ```

   b. Press `i` to enter edit mode, set `"multiNodesInferEnabled"=true` to enable multi-node inference, and modify the parameters in Table 6 as needed. For details, see [Configuration Parameters (Serving)](service_parameter_configuration.md).

      Table 6 Multi-node inference configuration

   | Configuration Item                | Configuration Description                                                    |
   | ---------------------- | ------------------------------------------------------------ |
   | multiNodesInferPort    | Port number for cross-node communication.                                          |
   | interNodeTLSEnabled    | Whether to enable certificate security authentication for cross-node communication. `true`: enabled. `false`: disabled. In this case, ignore the following parameters.|
   | interNodeTlsCaPath     | Path to the root certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCaFiles    | Root certificate name list. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCert       | Path to the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.        |
   | interNodeTlsPk         | Path to the private key file of the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.    |
   | interNodeTlsCrlPath    | Path to the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.|
   | interNodeTlsCrlFiles   | Name of the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.  |

   > [!NOTE]
   > - If HTTPS communication is disabled (·"httpsEnabled": false·), high network security risks exist.
   > - The `modelWeightPath` config file (`config.json`) must be owned by the current user (user/group match), not be a symlink, and have permissions no stricter than `640`. Otherwise, startup will fail.
   > - In a data center, if cross-node communication security authentication does not need to be enabled, set `interNodeTLSEnabled` to `false`. Disabling this option introduces significant network security risks.

   c. Press `Esc`, type `:wq!`, then press `Enter` to save and exit editing.

5. (Optional) If gRPC two-way authentication is enabled (that is, interNodeTLSEnabled is set to true),

    a. Import the certificate. [Table 7](#table7) describes the certificate files.

      > [!NOTE]
      > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
      > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
      > - When importing certificates, ensure that the permissions required by the CA certificate tool, service certificate tool, private key certificate tool, and CRL tool is 600, 600, 400, and 600, respectively.
      > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

    Table 7 Certificate file information <a id="table7"></a>

    | Certificate File              | Default Path                       | Description                                                        |
    | ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
    | Root certificate                | mindie-service/security/grpc/ca/    | Required when `interNodeTLSEnabled` is set to `true`.                    |
    | Service certificate              | mindie-service/grpc/certs/          | This parameter is mandatory when `interNodeTLSEnabled` is set to `true`.                    |
    | Private key of a service certificate          | mindie-service/security/grpc/keys/  | Private key file encryption is supported. Required when `interNodeTLSEnabled` is set to `true`.|
    | Service CRL      | mindie-service/security/grpc/certs/ | Required.                                                      |

    b. Run the following command in `{MindIE installation directory}/latest` to change the user permission on the certificate file:

    ```shell
    chmod 400 mindie-service/security/grpc/ca/*
    chmod 400 mindie-service/security/grpc/certs/*
    chmod 400 mindie-service/security/grpc/keys/*
    ```

6. (Optional) Enable HTTPS authentication (that is, set `httpsEnabled` to `true`).

    a. Import the certificate. For details about the certificate information, see [Table 8] (#table8).

    > [!NOTE]
    > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
    > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
    > - When importing certificates, ensure that the script permissions required for CA certificates, service certificates, private key certificates, and CRL certificates are 600, 400, and 600, respectively.
    > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

      Table 8 Certificate files <a id="table8"></a>

      | Certificate File              | Default Path                                            | Description                                             |
      | ---------------------- | -------------------------------------------------------- | ------------------------------------------------- |
      | Root certificate                | {MindIE installation directory}/latest/mindie-service/security/ca/   | Multiple CA certificates are supported.<br>Required when HTTPS is enabled.           |
      | Service certificate              | {MindIE installation directory}/latest/mindie-service/security/certs/| Required when HTTPS is enabled.                                |
      | Private key of a service certificate          | {MindIE installation directory}/latest/mindie-service/security/keys/ | Private key file encryption is supported.<br>Required when HTTPS is enabled.     |
      | Service CRL      | {MindIE installation directory}/latest/mindie-service/security/certs/  | Optional when HTTPS is enabled.                                |

    b. Run the following command in the `{MindIE installation directory}` to modify the user permission on the certificate files:

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
      ```

7. Configure environment variables.

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh                           # CANN
      source /usr/local/Ascend/nnal/atb/set_env.sh                                 # ATB
      source /usr/local/Ascend/atb-models/set_env.sh         # ATB Models
      ```

8. Copy the model weight file (prepared by yourself) to the directory specified by `modelWeightPath` in `config.json`.

      ```bash
      cp -r {Path_to_the_model_weight_file} {modelWeightPath}
      ```

9. Load environment variables.

    ```bash
    source mindie-service/set_env.sh
    ```

10. Set the environment variables `RANK_TABLE_FILE` and `MIES_CONTAINER_IP` (e.g., using the rank table example from the [sample ranktable file](https://gitcode.com/Ascend/MindIE-Motor-CPP/blob/v3.0.0/docs/en/user_guide/service_deployment/pd_separation_service_deployment.md); see Table 4 for details).

    - Container corresponding to the master node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Master node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

    - Container corresponding to the Slave node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Slave node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

11. Start the service by running the startup command in the ```/{MindIE installation directory}/latest/mindie-service``` directory. This operation must be performed in containers on both the master and slave nodes.

    - (Recommended) Start the service in background process mode.

      ```bash
      nohup ./bin/mindieservice_daemon > output.log 2>&1 &
      ```

      If the following information is displayed, the service is started successfully.

      ```text
      Daemon start success!
      ```

    - Start the service directly.

      ```bash
      ./bin/mindieservice_daemon
      ```

      If the following information is displayed, the service is started successfully.

      ```text
      Daemon start success!
      ```

> [!NOTE]
>
> - In the directory where the service is started, Ascend-CANN-Toolkit generates a `kernel_meta_temp_*xxxx*` directory to store the CCE files of operators. Therefore, start the inference service in a directory where the current user has write permissions, such as the `Ascend-mindie-server_{version}_linux-{arch}_{abi}` directory, or a temporary directory manually created under `Ascend-mindie-server_{version}_linux-{arch}`.
> - To switch to another user, run the `rm -f /dev/shm/*` command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
> - The `output.log` file captured by the standard output stream supports user-defined files and paths.
> - If service startup fails due to missing `lib*.so` dependencies, refer to "`libboost_thread.so.1.82.0` Cannot Be Found When MindIE Motor Is Started".
> - You are not advised to repeatedly start the service in the same container. Before repeatedly starting the service, delete the `*llm_backend_*` and `llm_tokenizer_shared_memory_*` files in the `/dev/shm/` directory of the container. The following commands are used as an example:

   ```bash
   find /dev/shm -name '*llm_backend_*' -type f -delete
   find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
   ```

### Procedure (using the `.run` package)

> [!NOTE]NOTE
> Perform the following operations on both the Master and Slave nodes.

1. Create and start a Docker container. The following uses the 8-card Ascend environment as an example.

   The following startup commands are for reference only. You can modify commands as required.

    ```bash
       docker run -it -d --net=host --shm-size=1g \
       --name container_name \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device=/dev/devmm_svm \
       --device=/dev/davinci0 \
       --device=/dev/davinci1 \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci4 \
       --device=/dev/davinci5 \
       --device=/dev/davinci6 \
       --device=/dev/davinci7 \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       -v /path-to-weights:/path-to-weights:ro \
       mindie:3.0.0-800I-A2-aarch64
   ```

2. Go to the MindIE installation directory as the installation user.

    ```bash
    cd {MindIE installation directory}
    ```

3. Check whether the directory/file permissions are the same as those shown in the following. If no, run the corresponding commands to modify the permissions.

    ```bash
   chmod 750 mindie-service
   chmod -R 550 mindie-service/bin
   chmod -R 500 mindie-service/bin/mindie_llm_backend_connector
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
    > If the file permission does not meet the requirements, the Server will fail to be started.

4. Set parameters in the container as required.

   Before the configuration, see the note in Step 3.

   a. Go to the `conf` directory and open the `config.json` file.

      ```bash
      cd ../conf
      vim config.json
      ```

   b. Press `i` to enter edit mode, set `"multiNodesInferEnabled"=true` to enable multi-node inference, and modify the parameters in Table 6 as needed. For details, see [Configuration Parameters (Serving)](service_parameter_configuration.md)..

      Table 6 Multi-node inference configuration

   | Configuration Item                | Configuration Description                                                    |
   | ---------------------- | ------------------------------------------------------------ |
   | multiNodesInferPort    | Port number for cross-node communication.                                          |
   | interNodeTLSEnabled    | Whether to enable certificate security authentication for cross-node communication. `true`: enabled. `false`: disabled. In this case, ignore the following parameters.|
   | interNodeTlsCaPath     | Path to the root certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCaFiles    | Root certificate name list. This parameter takes effect when `"interNodeTLSEnabled"=true`.            |
   | interNodeTlsCert       | Path to the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.        |
   | interNodeTlsPk         | Path to the private key file of the service certificate. This parameter takes effect when `"interNodeTLSEnabled"=true`.    |
   | interNodeTlsCrlPath    | Path to the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.|
   | interNodeTlsCrlFiles   | Name of the service certificate revocation list. This parameter takes effect when `"interNodeTLSEnabled"=true`.  |

   > [!NOTE]NOTE
   > - If HTTPS communication is disabled (·"httpsEnabled": false·), high network security risks exist.
   > - The `modelWeightPath` config file (`config.json`) must be owned by the current user (user/group match), not be a symlink, and have permissions no stricter than `640`. Otherwise, startup will fail.
   > - In a data center, if cross-node communication security authentication does not need to be enabled, set `interNodeTLSEnabled` to `false`. Disabling this option introduces significant network security risks.

   c. Press `Esc`, type `:wq!`, then press `Enter` to save and exit editing.

5. (Optional) If gRPC two-way authentication is enabled (that is, interNodeTLSEnabled is set to true),

    a. Import the certificate. [Table 7](#table7) describes the certificate files.

      > [!NOTE]NOTE
      > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
      > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
      > - When importing certificates, ensure that the permissions required by the CA certificate tool, service certificate tool, private key certificate tool, and CRL tool is 600, 600, 400, and 600, respectively.
      > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

    Table 7 Certificate file information <a id="table7"></a>

    | Certificate File              | Default Path                       | Description                                                        |
    | ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
    | Root certificate                | mindie-service/security/grpc/ca/    | Required when `interNodeTLSEnabled` is set to `true`.                    |
    | Service certificate              | mindie-service/grpc/certs/          | This parameter is mandatory when `interNodeTLSEnabled` is set to `true`.                    |
    | Private key of a service certificate          | mindie-service/security/grpc/keys/  | Private key file encryption is supported. Required when `interNodeTLSEnabled` is set to `true`.|
    | Service CRL      | mindie-service/security/grpc/certs/ | Required.                                                      |

    b. Run the following command in `{MindIE installation directory}/latest` to change the user permission on the certificate file:

    ```shell
    chmod 400 mindie-service/security/grpc/ca/*
    chmod 400 mindie-service/security/grpc/certs/*
    chmod 400 mindie-service/security/grpc/keys/*
    ```

6. (Optional) Enable HTTPS authentication (that is, set `httpsEnabled` to `true`).

    a. Import the certificate. For details about the certificate information, see [Table 8] (#table8).

    > [!NOTE]NOTE
    > - When three-plane isolation is enabled for HTTPS, you are advised not to use the same security certificate for the HTTPS service plane and management plane. Using the same security certificate can cause high network security risks.
    > - You are advised not to use the same security certificate for HTTPS and gRPC. Using the same certificate may lead to significant network security risks.
    > - When importing certificates, ensure that the script permissions required for CA certificates, service certificates, private key certificates, and CRL certificates are 600, 400, and 600, respectively.
    > - If the certificate import times out, see [Starting the haveged Service](../install/faq_and_appendixes/starting_the_haveged_service.md).

      Table 8 Certificate files <a id="table8"></a>

      | Certificate File              | Default Path                                            | Description                                             |
      | ---------------------- | -------------------------------------------------------- | ------------------------------------------------- |
      | Root certificate                | {MindIE installation directory}/latest/mindie-service/security/ca/   | Multiple CA certificates are supported.<br>Required when HTTPS is enabled.           |
      | Service certificate              | {MindIE installation directory}/latest/mindie-service/security/certs/| Required when HTTPS is enabled.                                |
      | Private key of a service certificate          | {MindIE installation directory}/latest/mindie-service/security/keys/ | Private key file encryption is supported.<br>Required when HTTPS is enabled.     |
      | Service CRL      | {MindIE installation directory}/latest/mindie-service/security/certs/  | Optional when HTTPS is enabled.                                |

    b. Run the following command in the `{MindIE installation directory}` to modify the user permission on the certificate files:

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
      ```

7. Configure environment variables.

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh                           # CANN
      source /usr/local/Ascend/nnal/atb/set_env.sh                                 # ATB
      source /usr/local/Ascend/atb-models/set_env.sh         # ATB Models
      ```

8. Copy the model weight file (prepared by yourself) to the directory specified by `modelWeightPath` in `config.json`.

      ```bash
      cp -r {Path_to_the_model_weight_file} {modelWeightPath}
      ```

9. Load environment variables.

    ```bash
    source mindie-service/set_env.sh
    ```

10. Set the environment variables `RANK_TABLE_FILE` and `MIES_CONTAINER_IP` (e.g., using the rank table example from the [sample ranktable file](https://gitcode.com/Ascend/MindIE-Motor-CPP/blob/v3.0.0/docs/en/user_guide/service_deployment/pd_separation_service_deployment.md); see Table 4 for details).

    - Container corresponding to the master node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Master node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

    - Container corresponding to the Slave node

         ```bash
         export MIES_CONTAINER_IP=IP address of the Slave node
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

11. Start the service by running the startup command in the ```/{MindIE installation directory}/latest/mindie-service``` directory. This operation must be performed in containers on both the master and slave nodes.

    - (Recommended) Start the service in background process mode.

      ```bash
      nohup ./bin/mindieservice_daemon > output.log 2>&1 &
      ```

      If the following information is displayed, the service is started successfully.

      ```text
      Daemon start success!
      ```

    - Start the service directly.

      ```bash
      ./bin/mindieservice_daemon
      ```

      If the following information is displayed, the service is started successfully.

      ```text
      Daemon start success!
      ```

> [!NOTE]NOTE
>
> - In the directory where the service is started, Ascend-CANN-Toolkit generates a `kernel_meta_temp_*xxxx*` directory to store the CCE files of operators. Therefore, start the inference service in a directory where the current user has write permissions, such as the `Ascend-mindie-server_{version}_linux-{arch}_{abi}` directory, or a temporary directory manually created under `Ascend-mindie-server_{version}_linux-{arch}`.
> - To switch to another user, run the `rm -f /dev/shm/*` command to delete the shared files created by the previous user. This prevents inference failure in case the new user does not have the read and write permissions on the shared files created by the previous user.
> - The `output.log` file captured by the standard output stream supports user-defined files and paths.
> - If service startup fails due to missing `lib*.so` dependencies, refer to "`libboost_thread.so.1.82.0` Cannot Be Found When MindIE Motor Is Started".
> - You are not advised to repeatedly start the service in the same container. Before repeatedly starting the service, delete the `*llm_backend_*` and `llm_tokenizer_shared_memory_*` files in the `/dev/shm/` directory of the container. The following commands are used as an example:

   ```bash
   find /dev/shm -name '*llm_backend_*' -type f -delete
   find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
   ```
