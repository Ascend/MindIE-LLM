# KV Cache池化使用指导

## 介绍
在当前大语言模型推理系统中，KV Cache是广泛采用的缓存机制，Prefix Cache技术基于KV Cache缓存机制能够在命中缓存时显著减少Prefill阶段的计算耗时。然而，Prefix Cache默认仅使用片上显存，其容量有限，难以缓存大量的前缀信息。为此，KV Cache池化特性实现了存储层级的扩展，支持将更大容量的存储介质纳入前缀缓存池中，从而突破片上内存的容量限制。KV Cache池化特性能够有效提升Prefix Cache的命中率，显著降低大模型推理的成本。

## 使用介绍
KV Cache池化特性依赖于Prefix Cache特性。此外，通过在MindIE的config.json配置文件中`BackendConfig`部分的以下字段配置KV Cache池化特性：
```
"kvPoolConfig" : {"backend":"", "configPath":""}
```
配置说明：
- `backend`：指定使用的池化后端。
- `configPath`：池化后端所需要的配置文件路径。

在开启Prefix Cache特性的前提下，当配置了上述两个字段后，则开启KV Cache池化特性。不同的池化后端需要自行安装。

## 已支持的池化后端
#### Mooncake
<details>

##### 一、通过源码安装

**Step1:** 通过源码安装Mooncake，详情参考Mooncake官方Build Guide。Mooncake当前支持了NPU亲和的传输方式——Ascend Direct Transport（详情可参考Mooncake官方文档），编译安装时需要打开对应选项`USE_ASCEND_DIRECT`：

```shell
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
mkdir build
cd build
cmake -DUSE_ASCEND_DIRECT=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF ..
make -j
make install
```

**Step2:** 当前，Mooncake Ascend Direct Transport要求在编译安装完成后需要拷贝对应的编译产物到`make install`时的安装路径，以`/usr/local/lib/python3.11/site-packages/mooncake`为例：
```
cp mooncake-common/src/libmooncake_common.so /usr/local/lib/python3.11/site-packages/mooncake
cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/lib/python3.11/site-packages/mooncake
cp mooncake-store/src/libmooncake_store.so /usr/local/lib/python3.11/site-packages/mooncake
```

**Step3:** 当前，Mooncake Ascend Direct Transport也要求使用到`hccn.conf`，因此需要拷贝宿主机`/etc/hccn.conf`至容器内`/etc/hccn.conf`。

**Step4:** 验证是否安装成功，若没有报错信息则安装成功：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
mooncake_master --port 12345
```

##### 二、准备Mooncake Client配置文件

使用Mooncake Ascend Direct Transport需要自行创建Mooncake client的配置文件，可参考Mooncake Store官方配置说明和Mooncake官方Ascend Direct Transport说明，例如创建`mooncake.json`：

```
{
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "global_segment_size": 268435456,
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "master_server_ip:50051",
    "use_ascend_direct": true
}
```

特殊配置（参数说明以Mooncake官方为准）：
- `metadata_server`：配置为"P2PHANDSHAKE"。
- `protocol`：配置为"ascend"。
- `use_ascend_direct`：配置为"true"。

##### 三、在MindIE中使用

依照前文[Mooncake client配置](#二准备mooncake-client配置文件)中准备好后，将Mooncake client配置文件路径配置到[使用介绍](#使用介绍)中的`configPath`字段，`backend`字段配置为`mooncake`。接着在终端1中拉起mooncake master server：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
mooncake_master --port 12345 --eviction_high_watermark_ratio 0.8 --eviction_ratio 0.05 --rpc_thread_num 128
```

`eviction_high_watermark_ratio`和`eviction_ratio`属于驱逐策略的参数，详情参考Mooncake官方Eviction Policy说明。`rpc_thread_num`属于master处理client链接的并发数量，建议适当调高该配置以高效处理并发请求。

接着在终端2中拉起MindIE服务，需要配置以下环境变量：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.11/site-packages/mooncake
export ASCEND_BUFFER_POOL=4:8
```

配置说明可参考Mooncake官方Ascend Direct Transport Important Notes。

</details>

## 声明

- 本代码仓提到的不同池化后端仅作为示例，仅供您用于非商业目的。如您使用这些池化后端完成示例，请您特别注意应遵守对应池化后端的License，如您因使用池化后端而产生侵权纠纷，华为不承担任何责任。
- 如您使用Mooncake池化后端，应悉知当前Mooncake仅支持Mooncake Master Server和Client之间以明文方式传输。在生产环境部署时，应确保相关IP与端口不暴露至公网，并限制访问到可信网络范围，以降低潜在的安全风险。
