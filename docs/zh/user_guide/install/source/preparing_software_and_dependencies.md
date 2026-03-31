# 准备软件包和依赖

介绍安装MindIE前，需要准备的软件包和依赖。

## 准备软件包

容器或裸机方式需要准备的软件包如如下：

|软件类型|软件包名称|软件说明|获取链接|
|--|--|--|--|
|MindIE LLM|mindie_llm-<*version>*-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|MindIE LLM组件安装包。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|ATB-Model|atb_llm-<*version>*-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|模型库安装包。使用MindIE LLM组件时，需要安装。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|MindIE Motor|mindie_motor-<*version>*-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|MindIE Motor组件安装包。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|MindIE SD|mindiesd-<*version>*-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|MindIE SD组件安装包。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-toolkit_<*version>*_linux-<*arch>*.run|CANN开发套件包（Toolkit）。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-<*chip_type>*-ops_<*version>*_linux-<*arch>*.run|CANN二进制算子包（ops）。<br> 安装ops前，需已安装同一版本胡Toolkit软件包，请选择运行设备对应的ops软件包。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-nnal_<*version>*_linux-<*arch>*.run|CANN神经网络加速库（NNAL）。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|Ascend Extension for PyTorch|torch_npu-<*torch_version>*.post<*post_id>*-cp*xxx*-cp*xxx*-manylinux_<*arch>*.whl|torch_npu插件whl包。|[获取链接](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)<br>在PyTorch栏单击对应版本后方“获取源码”按钮，跳转至PyTorch的gitcode仓库发布页，然后在页面下方获取对应版本的torch_npu。|
|Ascend Extension for PyTorch|apex-<*apex_version>*_ascend-cp*xxx*-cp*xxx*-<*arch*>.whl|APEX模块的whl包。|请参见《Ascend Extension for PyTorch 软件安装指南》中的“[安装APEX模块](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/installing_apex.md)”章节，根据Python3.10版本自行编译。|
|Ascend Extension for PyTorch|torch-<*torch_version>*+cpu-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|PyTorch框架whl包。推荐使用2.1.0版本。|请从《Ascend Extension for PyTorch 软件安装指南》中的“[安装PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)”章节获取。|

> [!NOTE]说明
>
> - <*version>*、<*torch_version*>和<*apex_version*>表示软件版本号
> - <*arch*>表示CPU架构
> - <*chip_type*>表示处理器类型
> - <*abi*>表示ABI版本

为了防止软件包在传递过程或存储期间被恶意篡改，下载软件包时需下载对应的数字签名文件用于完整性验证。

请单击[PGP数字签名工具包](https://support.huawei.com/enterprise/zh/tool/pgp-verify-TL1000000054)获取工具包，将工具包解压后，请参考文件夹中的《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请不要使用该软件包，访问[支持与服务](https://www.hiascend.com/support)在论坛求助或提交技术工单。

## 准备依赖

MindIE所需依赖如[表1](#table1)所示。

> [!NOTE]说明 
> 针对用户自行安装的开源软件，请使用稳定版本（尽量使用无漏洞的版本）。

**表 1**  依赖列表 <a id="table1"></a>

|软件|版本要求|变更记录|
|--|--|--|
|glibc|<li>Ascend-mindie\_<*version*>\_linux-\<*arch*>\_abi0.run配套的glibc版本需大于或等于2.34。</li><li>Ascend-mindie_<*version*>_linux-<*arch*>_abi1.run配套的glibc版本需大于或等于2.38。</li>|Mind 2.1.RC1版本修改|
|gcc、g++|大于或等于11.4.0，请用户自行安装。|Mind 1.0版本新增|
|Python|3.10.*x*、3.11.*x*|Mind 1.0版本新增|
|gevent|22.10.2|Mind 1.0版本新增|
|python-rapidjson|大于或等于1.6|Mind 1.0版本新增|
|geventhttpclient|2.0.11|Mind 1.0版本新增|
|urllib3|2.1.0|Mind 1.0版本新增|
|greenlet|3.0.3|Mind 1.0版本新增|
|zope.event|5.0|Mind 1.0版本新增|
|zope.interface|6.1|Mind 1.0版本新增|
|prettytable|3.5.0|Mind 1.0版本新增|
|jsonschema|4.21.1|Mind 1.0版本新增|
|jsonlines|4.0.0|Mind 1.0版本新增|
|thefuzz|0.22.1|Mind 1.0版本新增|
|pyarrow|大于或等于15.0.0|Mind 1.0版本新增|
|pydantic|2.6.3|Mind 1.0版本新增|
|sacrebleu|2.4.2|Mind 1.0版本新增|
|rouge_score|0.1.2|Mind 1.0版本新增|
|pillow|10.3.0|Mind 1.0版本新增|
|requests|2.31.0|Mind 1.0版本新增|
|matplotlib|大于或等于1.3.0|Mind 1.0版本新增|
|text_generation|0.7.0|Mind 1.0版本新增|
|numpy|1.26.3|Mind 1.0版本新增|
|pandas|2.1.4|Mind 1.0版本新增|
|transformers|4.39.3，请用户根据模型选择对应版本。|Mind 1.0版本新增|
|tritonclient[all]|-|Mind 1.0版本新增|
|numba|0.61.2|MindIE 2.0.RC1版本新增|
|posix_ipc|1.2.0|MindIE 2.2.RC1版本新增|
|fastapi|0.115.11|MindIE 2.3.0版本新增|
|uvicorn|0.34.3|MindIE 2.3.0版本新增|
