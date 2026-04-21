# PD混合部署

## 单机混部

### 前提条件

- 服务器或容器环境上已经安装好NPU驱动和固件、CANN包、PyTorch、ATB Models和MindIE。
- 若开启HTTPS双向认证，需要提前准备好服务证书、服务器私钥和验签证书等。
- 若使用容器化部署启动，要求共享内存设置不小于1GB。
- Server对于Python的环境要求为Python3.11.x。

### 操作步骤

1. 以安装用户进入MindIE安装目录。

    ```bash
     cd {MindIE安装目录}
     ```

2. 确认目录文件权限是否如下所示，若存在不匹配项，则参考以下命令修改权限。

    ```bash
    chmod 640 mindie_llm/conf/config.json
    ```

    > [!NOTE]说明
    > 若文件权限不符合要求将会导致Server启动失败。

3. 根据用户需要设置配置参数。

   配置前注意事项如下所示：

   | 参数名称              | 说明                                             | 注意事项                                                     |
   | --------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
   | httpsEnabled          | 开启HTTPS通信（即“httpsEnabled”=false时）        | 不开启，会存在较高的网络安全风险                             |
   | maxLinkNum            | 默认值为1000，推荐设置为300                      | 1000并发能力受模型性能影响，受限支持；一般在较小模型、较低序列长度下才可以使用1000并发 |
   | MIES_CONFIG_JSON_PATH | 用户可通过设置该环境变量提供此Server的配置文件   | 需要用户自行保障此配置文件的安全性                           |
   | modelWeightPath       | 模型权重路径，此路径下的所有文件由用户自行提供   | 需要用户自行保障此处所有文件的安全性；且该路径下的config.json文件需保证其用户组和用户名与当前用户一致，并且为非软链接，文件权限不高于750，若不符合要求将会导致Server启动失败 |
   | tlsCaFile             | 业务面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | tlsCert               | 业务面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | tlsPk                 | 业务面RESTful接口使用的服务证书私钥文件          | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | tlsCrlFiles           | 业务面RESTful接口使用的吊销列表文件列表          | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | managementTlsCaFile   | 管理面RESTful接口使用的CA证书文件列表            | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | managementTlsCert     | 管理面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | managementTlsPk       | 管理面RESTful接口使用的服务证书私钥文件          | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | managementTlsCrlFiles | 管理面RESTful接口使用的吊销列表文件列表          | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interCommTlsCaFiles   | PD分离场景下，PD节点间通信使用的CA证书文件列表   | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interCommTlsCert      | PD分离场景下，PD节点间通信使用的服务证书文件     | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | interCommPk           | PD分离场景下，PD节点间通信使用的服务证书私钥文件 | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | interCommTlsCrlFiles  | PD分离场景下，PD节点间通信使用的吊销列表文件列表 | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interNodeTlsCaFiles   | 多机场景下，主从节点间通信使用的CA证书文件列表   | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interNodeTlsCert      | 多机场景下，主从节点间通信使用的服务证书文件     | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | interNodeTlsPk        | 多机场景下，主从节点间通信使用的服务证书私钥文件 | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | interNodeTlsCrlFiles  | 多机场景下，主从节点间通信使用的吊销列表文件列表 | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |

   a. 进入conf目录，打开“config.json”文件。

    ```bash
    cd mindie_llm/conf
    vim config.json
    ```

   b. 按“i”进入编辑模式，根据用户需要修改配置参数，参数详情请参见[配置参数说明（服务化）](service_parameter_configuration.md)”章节。

   c. 按“Esc”键，输入`:wq!`，按“Enter”保存并退出编辑。

4. （可选）若开启了HTTPS认证（即“httpsEnabled” : true时，默认开启）。

   a. 使用Service的证书导入脚本导入证书，各证书信息如[表1](#table1)所示。

    > [!NOTE]说明
    > - HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
    > - HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
    > - 导入证书时，对于用户导入CA证书的脚本权限要求为600，服务证书的脚本权限要求为600，私钥证书的脚本权限要求为400，吊销列表证书的脚本权限要求为600。
    > - Service的证书导入脚本请参见《MindIE Motor开发指南》中的“配套工具 > MindIE Service Tools > <a href="https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindieservice/servicedev/mindie_service0312.html">CertTools</a>”章节。
    > - 如果导入证书超时，请参考[启动haveged服务](../install/faq_and_appendixes/starting_the_haveged_service.md)处理。

      表1 证书文件清单  <a id="table1"></a>

      | 证书文件               | 默认目标路径                                             | 说明                                              |
      | ---------------------- | -------------------------------------------------------| ------------------------------------------------- |
      | 根证书                 | {MindIE安装目录}/latest/mindie-service/security/ca/    | 支持多个CA证书。<br/><br/>开启HTTPS后必选。       |
      | 服务证书               | {MindIE安装目录}/latest/mindie-service/security/certs/ | 开启HTTPS后必选。                                 |
      | 服务证书私钥            | {MindIE安装目录}/latest/mindie-service/security/keys/  | 支持私钥文件加密场景。<br/><br/>开启HTTPS后必选。 |
      | 服务证书吊销列表        | {MindIE安装目录}/latest/mindie-service/security/certs/  | 开启HTTPS后可选。                                 |
      | 服务证书私钥的加密口令  | {MindIE安装目录}/latest/mindie-service/security/pass/   | 可选。                                            |

   b. 在{MindIE安装目录}下执行以下命令修改证书文件的用户权限。

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
        chmod 400 mindie-service/security/pass/*
      ```

5. 使用以下命令配置环境变量。

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh                                 # CANN
    source /usr/local/Ascend/nnal/atb/set_env.sh                                       # ATB
    source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh               # ATB Models
    ```

6. 将模型权重文件（由用户自行准备）拷贝到config.json中模型配置参数“modelWeightPath”指定的目录下。

    ``` shell
    cp -r {模型权重文件所在路径} {modelWeightPath}
    ```

7. 配置ATB Models的运行环境变量。

    ```shell
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

8. 启动服务。

    > [!NOTE]说明
    > 拉起服务前，建议用户使用MindStudio的预检工具进行配置文件字段校验，辅助校验配置的合法性，详情请参见[链接](https://gitcode.com/Ascend/msit/tree/master/msprechecker)。

    直接启动服务。

    ```shell
    mindie_llm_server
    ```

     回显如下则说明启动成功。

    ```text
    Daemon start success!
    ```

 > [!NOTE]说明
 >
 > - Ascend-cann-toolkit工具会在执行服务启动的目录下生成kernel_meta_temp_xxxx目录，该目录为算子的cce文件保存目录。因此需要在当前用户拥有写权限目录下（例如Ascend-mindie-server_{version}_linux-{arch}_{abi}目录，或者用户在Ascend-mindie-server_{version}_linux-{arch}目录下自行创建临时目录）启动推理服务。
 > - 如需切换用户，请在切换用户后执行rm -f /dev/shm/*命令，删除由之前用户运行创建的共享文件。避免切换用户后，该用户没有之前用户创建的共享文件的读写权限，造成推理失败。
 > - 标准输出流捕获到的文件output.log支持用户自定义文件和路径。
 > - 服务启动报缺失lib*.so依赖的错误时，处理方法请参见启动MindIE Motor服务时，出现找不到libboost_thread.so.1.82.0报错章节。
 > - 不建议在同一容器中反复拉起服务，重复拉起前请清理容器“/dev/shm/”目录下的*llm_backend_*和llm_tokenizer_shared_memory_*文件，参考命令如下：
 >
>      ```shell
>      find /dev/shm -name '*llm_backend_*' -type f -delete
>      find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
>      ```

## 多机混部

单个模型权重过大，单台推理机显存有限，无法容纳整个模型权重参数时，需要采用多个节点进行多机推理。

### 前提条件

- Server对于Python的环境要求为Python3.11.x。此处以Python3.11为例，如果环境中的Python3.11不是默认版本，需要参考如下方法添加环境变量（Python路径根据实际路径进行修改）。

    ```linux
    export LD_LIBRARY_PATH=/usr/local/python3.11/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/python3.11/bin:$PATH
    ```

- 服务器或容器环境上已经安装好NPU驱动和固件、CANN包、PyTorch、ATB Models和MindIE。

- 若使用容器化部署启动，要求共享内存设置不小于1GB。

- 若开启HTTPS双向认证或多机通信认证，需要提前准备好服务证书、服务器私钥、验签证书等，详情请参见《MindIE Motor开发指南》中的“集群服务部署 > 单机（非分布式）服务部署 > 安装部署 > 使用Deployer部署服务示例 > 部署Deployer服务端 > [准备TLS证书](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/service_deployment/single_machine_service_deployment.md)”章节。

### 使用限制

- 仅支持Atlas 800I A2 推理服务器环境，最大支持4机32卡的多机推理，多机推理支持的模型请参见[模型列表](../model_support_list.md)；不支持Atlas 300I Duo 推理卡环境。
- “maxLinkNum”默认值为1000，推荐设置为300。1000并发能力受模型性能影响受限支持，一般较小模型、较低序列长度下才可以使用1000并发。
- 不同节点的权重的默认采样参数需要配置一致，否则在没有配置采样参数的情况下，推理服务可能卡死。

### 相关环境变量

| 参数名称              | 参数说明                                                     |
| --------------------- | ------------------------------------------------------------ |
| MIES_CONTAINER_IP     | 容器部署时，请设置成容器的IP地址（如果容器与裸机共用IP地址，应当配置为裸机IP地址），会用于多机间gRPC（Google Remote Procedure Call）通信和EndPoint业务面接收请求。裸机部署时，不配置。 |
| HOST_IP               | 裸机部署时（不建议使用裸机部署），请设置成机器的物理机或虚拟机的IP地址。容器部署时不配置。 |
| RANK_TABLE_FILE       | ranktable.json文件的绝对路径。  多机推理必须配置。 单机推理建议取消该环境变量（取消命令：**unset RANK_TABLE_FILE**）。如果设置该环境变量，文件内容必须正确有效（节点IP地址和device_ip必须正确），否则会导致模型初始化失败。 |
| MIES_CONFIG_JSON_PATH | config.json文件的路径。如果该环境变量存在，则读取该值。如果不存在，则读取${MINDIE_LLM_HOME_PATH}/conf/config.json文件。 |
| HCCL_DETERMINISTIC    | HCCL通信的确定性计算。多机推理时，建议配置为true。           |

> [!NOTE]说明
> Server启动时，会根据“multiNodesInferEnabled”参数判断是单机推理还是多机推理：
>
> - “multiNodesInferEnabled” : false代表单机推理，Server在启动过程中不会读取“RANK_TABLE_FILE”环境变量，但是底层模型加速库初始化时，会尝试读取该环境变量。所以在单机推理场景中，如果设置了该环境变量，请保证文件内容值的正确性（即：server_count=1、节点IP、device_ip和rank_id等必须正确）。
> - “multiNodesInferEnabled” : true代表多机推理，
>   - Server在启动过程中，会读取“RANK_TABLE_FILE”环境变量，并判断ranktable文件内容是否有效。
>   - 当开启多机推理时，config.json中的“npuDeviceIds”和“worldSize”将失效，具体使用卡号及总体Rank数，将根据ranktable文件确定。
> - rank_id=0的节点为Master节点，其余为Slave节点。
> - Master服务实例，可以接收用户推理请求；Slave实例无法接收用户推理请求。

### ranktable文件样例

ranktable.json文件权限需要设置为640，详细内容请根据以下样例进行配置。（该文件需要用户自行编写）

   ```json
   {
      "version": "1.0",
      "server_count": "2",
      "server_list": [
         {
               "server_id": "Master节点IP地址",
               "container_ip": "Master节点容器IP地址",
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
               "server_id": "Slave节点IP地址",
               "container_ip": "Slave节点容器IP地址",
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

参数说明：

- Master/Slave节点IP地址：请根据实际情况进行修改。
- Master/Slave节点容器IP地址：一般与Master/Slave节点IP地址一致，如果启动容器时使用了**--net=host**，则需要与宿主机IP地址一致，请根据实际情况进行修改。
- device_id：表示在实际节点上的第几个NPU设备。
- device_ip：表示NPU设备的IP地址，可通过hccn_tool进行配置。
- rank_id：表示推理进程Rank编号。

> [!NOTE]说明
> ranktable.json文件通过环境变量“RANK_TABLE_FILE”配置，若用户自行提供此文件，需要用户自行保障此配置文件的安全性，且Master节点和Slave节点上都需要创建该文件。

### 操作步骤

> [!NOTE]说明
> Master节点和Slave节点上均需执行以下操作。

1. 创建并启动Docker容器，此处以8卡昇腾环境为例。

   以下启动命令仅供参考，可根据需求自行修改。

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
       mindie:2.2.RC1-800I-A2-aarch64
   ```

2. 以安装用户进入MindIE安装目录。

    ```bash
    cd {MindIE安装目录}
    ```

3. 确认目录文件权限是否如下所示，若存在不匹配项，则参考以下命令修改权限。

    ```bash
    chmod 640 mindie_llm/conf/config.json
    ```

    > [!NOTE]说明
    > 若文件权限不符合要求将会导致Server启动失败。

4. 在容器中，根据用户需要设置配置参数。

   配置前请参见3中的注意事项。

   a. 进入conf目录，打开“config.json”文件。

      ```bash
      cd mindie_llm/conf
      vim config.json
      ```

   b. 按“i”进入编辑模式，设置“multiNodesInferEnabled”=true开启多机推理，并根据用户需要修改表1的参数，参数详情请参见[配置参数说明（服务化）](service_parameter_configuration.md)。

      表1 多机推理相关配置

   | 配置项                 | 配置说明                                                     |
   | ---------------------- | ------------------------------------------------------------ |
   | multiNodesInferPort    | 跨机通信的端口号。                                           |
   | interNodeTLSEnabled    | 跨机通信是否开启证书安全认证。true：开启证书安全认证。false：关闭证书安全认证。若关闭证书安全认证，可忽略以下参数。 |
   | interNodeTlsCaPath     | 根证书名称路径。“interNodeTLSEnabled”=true生效。             |
   | interNodeTlsCaFiles    | 根证书名称列表。“interNodeTLSEnabled”=true生效。             |
   | interNodeTlsCert       | 服务证书文件路径。“interNodeTLSEnabled” : true生效。         |
   | interNodeTlsPk         | 服务证书私钥文件路径。“interNodeTLSEnabled” : true生效。     |
   | interNodeTlsCrlPath    | 服务证书吊销列表文件夹路径。“interNodeTLSEnabled”=true生效。 |
   | interNodeTlsCrlFiles   | 服务证书吊销列表名称列表。“interNodeTLSEnabled”=true生效。   |

   > [!NOTE]说明
   > - 如果不开启HTTPS通信（即“httpsEnabled” : false时），会存在较高的网络安全风险。
   > - “modelWeightPath”参数配置路径下的config.json文件，需保证其用户组和用户名与当前用户一致，并且为非软链接，文件权限不高于640，若不符合要求将会导致启动失败。
   > - 在数据中心内部，如果不需要开启跨机通信安全认证，请配置“interNodeTLSEnabled” : false，若关闭跨机通信安全认证（即“interNodeTLSEnabled” : false），会存在较高的网络安全风险。

   c. 按“Esc”键，输入`:wq!`，按“Enter”保存并退出编辑。

5. （可选）若开启了GRPC双向认证（即“interNodeTLSEnabled”=true时）。

    a. 使用证书管理工具导入证书，证书管理工具的使用请参见《MindIE Motor开发指南》中的“配套工具 > MindIE Service Tools > [CertTools](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/service_deployment/single_machine_service_deployment.md)”章节。各证书文件信息如[表2](#table2)所示。

      > [!NOTE]说明
      > - HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
      > - HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
      > - 导入证书时，对于用户导入的CA文件证书工具要求的权限为600，服务证书文件证书工具要求的权限为600，私钥文件证书工具要求的权限要求为400，吊销列表证书工具要求的权限为600。
      > - 如果导入证书超时，请参考[启动haveged服务](../install/faq_and_appendixes/starting_the_haveged_service.md)处理。

    表2 证书文件信息    <a id="table2"></a>

    | 证书文件               | 默认目标路径                        | 说明                                                         |
    | ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
    | 根证书                 | mindie-service/security/grpc/ca/    | 开启“interNodeTLSEnabled” : true后必选。                     |
    | 服务证书               | mindie-service/grpc/certs/          | 开启“interNodeTLSEnabled” : true后必选。                     |
    | 服务证书私钥           | mindie-service/security/grpc/keys/  | 支持私钥文件加密场景。开启“interNodeTLSEnabled” : true后必选。 |
    | 服务证书吊销列表       | mindie-service/security/grpc/certs/ | 必选。                                                       |
    | 服务证书私钥的加密口令 | mindie-service/security/pass/       | 必选。                                                       |

    b. 在{MindIE安装目录}/latest下执行以下命令修改证书文件的用户权限。

    ```shell
    chmod 400 mindie-service/security/grpc/ca/*
    chmod 400 mindie-service/security/grpc/certs/*
    chmod 400 mindie-service/security/grpc/keys/*
    chmod 400 mindie-service/security/grpc/pass/*
    ```

6. （可选）若开启了HTTPS认证（即“httpsEnabled”: true时，默认开启）。

    a. 使用Service的证书导入脚本导入证书，各证书信息如[表3](#table3)所示。

    > [!NOTE]说明
    > - HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
    > - HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
    > - 导入证书时，对于用户导入CA证书的脚本权限要求为600，服务证书的脚本权限要求为600，私钥证书的脚本权限要求为400，吊销列表证书的脚本权限要求为600。
    > - Service的证书导入脚本请参见《MindIE Motor开发指南》中的“配套工具 > MindIE Service Tools > CertTools”章节。
    > - 如果导入证书超时，请参考[启动haveged服务](../install/faq_and_appendixes/starting_the_haveged_service.md)处理。

      表3 证书文件清单  <a id="table3"></a>

      | 证书文件               | 默认目标路径                                             | 说明                                              |
      | ---------------------- | -------------------------------------------------------- | ------------------------------------------------- |
      | 根证书                 | {MindIE安装目录}/latest/mindie-service/security/ca/    | 支持多个CA证书。<br>开启HTTPS后必选。            |
      | 服务证书               | {MindIE安装目录}/latest/mindie-service/security/certs/ | 开启HTTPS后必选。                                 |
      | 服务证书私钥           | {MindIE安装目录}/latest/mindie-service/security/keys/  | 支持私钥文件加密场景。<br>开启HTTPS后必选。      |
      | 服务证书吊销列表       | {MindIE安装目录}/latest/mindie-service/security/certs/   | 开启HTTPS后可选。                                 |
    | 服务证书私钥的加密口令 | {MindIE安装目录}/latest/mindie-service/security/pass/  | 可选。                                  |

    b. 在{MindIE安装目录}下执行以下命令修改证书文件的用户权限。

      ```bash
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
        chmod 400 mindie-service/security/pass/*
      ```

7. 使用以下命令配置环境变量。

      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh                           # CANN
      source /usr/local/Ascend/nnal/atb/set_env.sh                                 # ATB
      source /usr/local/lib/python3.11/site-packages/mindie_llm/set_env.sh         # ATB Models
      ```

8. 将模型权重文件（由用户自行准备）拷贝到config.json中模型配置参数“modelWeightPath”指定的目录下。

      ```bash
      cp -r {模型权重文件所在路径} {modelWeightPath}
      ```

9. 配置ATB Models的运行环境变量。

    ```shell
    ATB_LLM_PATH=$(python3 -c "import atb_llm, os; print(os.path.dirname(atb_llm.__file__))")
    export ATB_SPEED_HOME_PATH=${ATB_LLM_PATH}
    export LD_LIBRARY_PATH=${ATB_LLM_PATH}/lib:${LD_LIBRARY_PATH}
    ```

10. 配置环境变量“RANK_TABLE_FILE”和“MIES_CONTAINER_IP”（以[ranktable文件样例](https://gitcode.com/Ascend/MindIE-Motor/blob/dev/docs/zh/user_guide/service_deployment/pd_separation_service_deployment.md)中的ranktable为例，具体参见表4）。

    - Master节点容器中

         ```bash
         export MIES_CONTAINER_IP=Master节点IP地址
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

    - Slave节点容器中

         ```bash
         export MIES_CONTAINER_IP=Slave节点IP地址
         export RANK_TABLE_FILE=${path}/ranktable.json
         export HCCL_DETERMINISTIC=true
         ```

11. 启动服务。此操作在Master节点容器和Slave节点容器中均需执行。

    - 直接启动。

      ```bash
      mindie_llm_server
      ```

      回显如下则说明启动成功。

      ```text
      Daemon start success!
      ```

> [!NOTE]说明
>
> - Ascend-cann-toolkit工具会在执行服务启动的目录下生成kernel_meta_temp_xxxx目录，该目录为算子的cce文件保存目录。因此需要在当前用户拥有写权限目录下（例如Ascend-mindie-server_{version}_linux-{arch}_{abi}目录，或者用户在Ascend-mindie-server_{version}_linux-{arch}目录下自行创建临时目录）启动推理服务。
> - 如需切换用户，请在切换用户后执行rm -f /dev/shm/*命令，删除由之前用户运行创建的共享文件。避免切换用户后，该用户没有之前用户创建的共享文件的读写权限，造成推理失败。
> - 标准输出流捕获到的文件output.log支持用户自定义文件和路径。
> - 服务启动报缺失lib*.so依赖的错误时，处理方法请参见启动MindIE Motor服务时，出现找不到libboost_thread.so.1.82.0报错章节。
> - 不建议在同一容器中反复拉起服务，重复拉起前请清理容器“/dev/shm/”目录下的*llm_backend_*和llm_tokenizer_shared_memory_*文件，参考命令如下：

   ```bash
   find /dev/shm -name '*llm_backend_*' -type f -delete
   find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete
   ```
