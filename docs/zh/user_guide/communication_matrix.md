# 通信矩阵

提供了MindIE LLM的通信矩阵，包括产品开放的端口、该端口使用的传输层协议、通过该端口与对端通信的通信网元名称、认证方式、用途等信息说明。

> [!NOTE]说明
> datadist、HCCL、LCCL和ATB相关通信矩阵请参考《[CANN 通信矩阵](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/commumatrix/commumatrix_01.html)》。

|源设备|源IP地址|源端口|目的设备|目的IP地址|目的端口（侦听）|协议|端口说明|侦听端口是否可更改|认证方式|加密方式|所属平面|版本|特殊场景|备注|
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
|MindIE LLM mempool-Mooncake模块|客户端通信IP地址|随机端口，由Mooncake分配，端口从8790开始。|Mooncake Master服务端|由用户拉起Mooncake Master Server时配置决定。|由用户拉起Mooncake Master Server时配置决定。|TCP|用于Mooncake池化client端与mooncake master server端通信。|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|MindIE LLM mempool-Mooncake模块|客户端通信IP地址|用户自定义配置|Mooncake Metadata服务端|由用户拉起Mooncake Master Server时配置决定。|由用户拉起Mooncake Master Server时配置决定。|TCP|用于Mooncake池化client端与mooncake metadata server端通信。|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|推理服务器|本机通信IIP地址|动态端口（1024~65535）|NPU|device侧IP地址|默认1024|TCP|hdk kmsagent组件监听端口，负责注册混淆因子。|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|NPU|device侧IP地址|默认1024|推理服务器|本机通信IIP地址|动态端口（1024~65535）|TCP|hdk kmsagent组件监听端口，负责注册混淆因子。|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|多机下的次节点客户端|客户端通信IP地址|环境变量"MASTER_PORT"配置|多机下的主节点客户端|推理服务环境变量MASTER_IP字段对应的IP地址（管理面和业务使用不同的IP地址场景）。|根据现网客户实际要求配置的固定端口，对应环境变量MASTER_PORT字段（管理面与业务面使用不同端口场景），可配范围为1024~65535，不配默认配置为None。|TCP|用pytorch的init_process_group函数发送请求，用于流水线并行不同机器之间的数据通信|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|多机下的主节点客户端|推理服务环境变量MASTER_IP字段对应的IP地址（管理面和业务使用不同的IP地址场景）。|根据现网客户实际要求配置的固定端口，对应环境变量MASTER_PORT字段（管理面与业务面使用不同端口场景），可配范围为1024~65535，不配默认配置为None。|多机下的次节点客户端|客户端通信IP地址|环境变量"MASTER_PORT"配置|TCP|用pytorch的init_process_group函数发送请求，用于流水线并行不同机器之间的数据通信|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|推理服务器|本机通信IP地址|随机端口|huggin face服务器|huggin face服务器ip|随机端口|不涉及|trust_remote_code可配置为true（默认为false），此时会使用权重文件夹中的代码文件，如果文件不存在可能触发下载。|不涉及|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|MindIE Sever服务端|部署的MindIE Sever服务端Pod IP|随机端口（由操作系统自动分配），默认范围32768~60999。|MindIE Sever服务端|部署的MindIE集群服务服务端Pod IP|随机端口（由操作系统自动分配），默认范围32768~60999。|TCP|MindIE Sever服务端的多DP进程间通信，共dp_size个port|否|无|无|数据面|MindIE 3.0.0|无|无|
|分布式边云协同数据传输服务端|推理服务config.json配置文件中layerwiseDisaggregatedMasterIpAddress字段对应的IP地址。|可配范围（1024~65535），默认配置为10000|分布式边云协同云侧（slave）设备|分布式边云协同云侧（slave）设备服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress的IP对应|随机端口（由操作系统自动分配）|TCP|分布式边云协同中边云hidden数据传输|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|分布式边云协同数据传输客户端|推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress字段对应的IP地址。|随机端口（由操作系统自动分配）|分布式边云协同边侧（master）设备|分布式边云协同边侧（master）设备服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedMasterIpAddress的IP对应|可配范围（1024~65535），默认配置为10000|TCP|分布式边云协同中边云hidden数据传输|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|分布式边云协同控制信号传输服务端|推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress字段对应的IP地址。|可配范围（1024~65535），默认配置为10001（prefill控制信号端口）、10002（decode控制信号端口）|分布式边云协同边侧（master）设备|分布式边云协同边侧（master）设备服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedMasterIpAddress的IP对应|随机端口（由操作系统自动分配）|TCP|分布式边云协同中边云TCP控制信号传输|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|分布式边云协同控制信号传输客户端|推理服务config.json配置文件中layerwiseDisaggregatedMasterIpAddress字段对应的IP地址。|随机端口（由操作系统自动分配）|分布式边云协同云侧（slave）设备|分布式边云协同云侧（slave）设备服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress的IP对应|可配范围（1024~65535），默认配置为10001（prefill控制信号端口）、10002（decode控制信号端口）|TCP|分布式边云协同中边云TCP控制信号传输|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|分布式边云协同多机云（slave）设备决策广播服务端（云master）|多机云设备的master对应的IP，与layerwiseDisaggregatedSlaveIpAddress中的首个地址对应|可配范围（1024~65535），默认配置为10003|分布式边云协同云侧（slave）设备的slave设备|分布式边云协同多机云设备的slave服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress的非首个地址对应|随机端口（由操作系统自动分配）|TCP|分布式边云协同多机决策（跨机）广播|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
|分布式边云协同多机云（slave）设备决策广播客户端（云slave）|多机云设备的master对应的IP，与layerwiseDisaggregatedSlaveIpAddress中的非首个地址对应|随机端口（由操作系统自动分配）|分布式边云协同云侧（slave）设备的master设备|分布式边云协同多机云设备的slave服务器IP，与推理服务config.json配置文件中layerwiseDisaggregatedSlaveIpAddress的首个地址对应|可配范围（1024~65535），默认配置为10003|TCP|分布式边云协同多机决策（跨机）广播|是|证书认证|TLS1.3|数据面|MindIE 3.0.0|无|无|
