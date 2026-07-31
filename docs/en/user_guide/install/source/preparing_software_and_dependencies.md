# Preparing Software Packages and Dependencies

The following describes the software packages and dependencies required for installing MindIE.

## Version Mapping

MindIE, CANN, and Ascend Extension for PyTorch must be used together. [Table 1](#table1) lists the version mapping.

**Table 1** Version mapping <a id="table1"></a>

|MindIE|CANN|Ascend Extension for PyTorch|
|-----------|------------|----------|
|3.0.0|8.5.1|7.3.0 (torch and torch_npu: 2.9.0) (recommended)<br> 7.2.0 (torch, torch_npu: 2.1.0)|

> [!NOTE]
> DeepSeek-V3.2 does not support torch and torch_npu 2.1.0.

## Software Package Preparation

### Using the `.whl` package

[Table 2](#table2) lists the software packages required for container or bare metal deployment.

**Table 2** Software packages <a id="table2"></a>

|Software Type|Package Name|Software Description|How to Obtain|
|--|--|--|--|
|MindIE LLM|mindie_llm-<*version>*-cp<*xxx>*-cp<*xxx>*-linux_<*arch>*.whl|MindIE LLM installation package.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|ATB-Model|atb_llm-<*version>*-cp<*xxx>*-cp<*xxx>*-linux_<*arch>*.whl|ATB Models installation package It is required when the MindIE LLM component is used.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|MindIE Motor|mindie_motor-<*version>*-cp<*xxx>*-cp<*xxx>*-linux_<*arch>*.whl|MindIE Motor installation package.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|MindIE SD|mindiesd-<*version>*-cp<*xxx>*-cp<*xxx>*-linux_<*arch>*.whl|MindIE SD installation package.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-toolkit_<*version>*_linux-<*arch>*.run|CANN development kit (Toolkit).|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-<*chip_type>*-ops_<*version>*_linux-<*arch>*.run|CANN binary operator package (ops).<br> Before installing the ops, the Toolkit software package of the same version must be installed. Select the ops software package corresponding to the running device.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-nnal_<*version>*_linux-<*arch>*.run|CANN neural network acceleration library (NNAL).|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|Ascend Extension for PyTorch|torch_npu-<*torch_version>*.post<*post_id>*-cp*xxx*-cp*xxx*-manylinux_<*arch>*.whl|WHL package of the torch_npu plugin|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)<ul><li>To obtain torch_npu v2.1.0, go to the community edition resource download page, select PyTorch 7.2.0 under the resources area in the top-left corner. </li><li>In the PyTorch column, click the download button next to your target version to go to the GitCode repository of PyTorch, and then download torch_npu.</li></ul>|
|Ascend Extension for PyTorch|apex-<*apex_version>*_ascend-cp*xxx*-cp*xxx*-<*arch*>.whl|WHL package of the APEX module.|Refer to [Installing APEX Module](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/installing_apex.md) in *Ascend Extension for PyTorch Software Installation Guide* and compile it based on Python 3.11.|
|Ascend Extension for PyTorch|torch-<*torch_version>*+cpu-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|WHL package of the PyTorch framework|<ul><li>For torch_npu 2.1.0, refer to [Installing PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html) in *Ascend Extension for PyTorch Software Installation Guide*. </li><li>For torch_npu 2.9.0, refer to [Installing PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) in *Ascend Extension for PyTorch Software Installation Guide*.</li></ul>|

### Using the `.run` package

[Table 3](#table3) lists the software packages required for container or bare metal deployment.

**Table 3** Software packages <a id="table3"></a>

|Software Type|Package Name|Software Description|How to Obtain|
|--|--|--|--|
|MindIE|Ascend-mindie_<*version>*\_linux-<*arch>*_\<abi>.run|Inference engine software package, which is used to develop applications based on MindIE.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|ATB-Model|Ascend-mindie-atb-models-<*version>*\_linux_<*arch>*_pyxxx_torchx.x.x-\<abi>.tar.gz|ATB Models installation package.This component needs to be installed when MindIE Motor and MindIE LLM are used.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-toolkit_<*version>*_linux-<*arch>*.run|CANN development kit (Toolkit).|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-<*chip_type>*-ops_<*version>*_linux-<*arch>*.run|CANN binary operator package (ops).<br> Before installing the ops, the Toolkit software package of the same version must be installed. Select the ops software package corresponding to the running device.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|CANN|Ascend-cann-nnal_<*version>*_linux-<*arch>*.run|CANN neural network acceleration library (NNAL)|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)|
|Ascend Extension for PyTorch|torch_npu-<*torch_version>*.post<*post_id>*-cp*xxx*-cp*xxx*-manylinux_<*arch>*.whl|WHL package of the torch_npu plugin.|[Download link](https://www.hiascend.com/developer/download/community/result?module=ie%2Bpt%2Bcann)<ul><li>To obtain torch_npu 2.1.0, select 7.2.0 from the PyTorch drop-down list in the Matching Resources area on the upper left of the page for downloading resources of the community edition.</li><li>In the PyTorch area, click the Obtain Source Code button next to your target version to go to the GitCode repository of PyTorch, and then download torch_npu.</li></ul>|
|Ascend Extension for PyTorch|apex-<*apex_version>*_ascend-cp*xxx*-cp*xxx*-<*arch*>.whl|WHL package of the Apex module.|Compile it based on Python3.10. For details, see [Installing the Apex Module](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/installing_apex.md) in Ascend Extension for PyTorch Software Installation Guide.|
|Ascend Extension for PyTorch|torch-<*torch_version>*+cpu-cp*xxx*-cp*xxx*-linux_<*arch>*.whl|WHL package of the PyTorch framework.|<ul><li>PyTorch framework and torch_npu plugin (2.1.0): Obtain it from "[Installing the PyTorch Framework](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html)"。</li><li>PyTorch framework and torch_npu plugin (2.6.0): Obtain it from "[Installing PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)" in Ascend Extension for PyTorch Software Installation Guide.</li></ul>|

> [!NOTE]
>
> - `<version>`, `<torch_version>`, and `<apex_version>` indicate software versions.
> - `<arch>` indicates the CPU architecture.
> - `<chip_type>` indicates the processor type.
> - `<abi>` indicates the ABI version.

Download the corresponding digital signature file for integrity verification when downloading the software package in case the software package is tampered with during transmission or storage.

Download [PGP Verify](https://support.huawei.com/enterprise/en/tool/pgp-verify-TL1000000054), decompress it, and verify the PGP digital signature of the downloaded package by referring to *OpenPGP Signature Verification Guide*. If the verification fails, do not use the package. Visit the [support website](https://www.hiascend.com/support) to get help from the community or submit a service ticket.

## Dependency Preparation

[Table 4](#table4) lists the dependencies required by MindIE.

> [!NOTE]
> Use stable versions (vulnerability-free versions are recommended) of open-source software.

**Table 4** Dependency list <a id="table4"></a>

|Software|Version Requirements|Change History|
|--|--|--|
|glibc|<li>The `Ascend-mindie_<version>_linux-<arch>_abi0.run` package requires `glibc >= 2.34`. </li><li>The `Ascend-mindie_<version>_linux-<arch>_abi1.run` package requires `glibc >= 2.38`.</li>|Modified in Mind 2.1.RC1|
|GCC, G++|11.4.0 or later. You need to install it by yourself.|Added in Mind 1.0|
|Python|3.11|Added in Mind 1.0|
|gevent|22.10.2|Added in Mind 1.0|
|python-rapidjson|1.6 or later|Added in Mind 1.0|
|geventhttpclient|2.0.11|Added in Mind 1.0|
|urllib3|2.1.0|Added in Mind 1.0|
|greenlet|3.0.3|Added in Mind 1.0|
|zope.event|5.0|Added in Mind 1.0|
|zope.interface|6.1|Added in Mind 1.0|
|prettytable|3.5.0|Added in Mind 1.0|
|jsonschema|4.21.1|Added in Mind 1.0|
|jsonlines|4.0.0|Added in Mind 1.0|
|thefuzz|0.22.1|Added in Mind 1.0|
|pyarrow|15.0.0 or later|Added in Mind 1.0|
|pydantic|2.6.3|Added in Mind 1.0|
|sacrebleu|2.4.2|Added in Mind 1.0|
|rouge_score|0.1.2|Added in Mind 1.0|
|pillow|10.3.0|Added in Mind 1.0|
|requests|2.31.0|Added in Mind 1.0|
|matplotlib|1.3.0 or later|Added in Mind 1.0|
|text_generation|0.7.0|Added in Mind 1.0|
|numpy|1.26.3|Added in Mind 1.0|
|pandas|2.1.4|Added in Mind 1.0|
|transformers|4.39.3. Select the corresponding version based on the model.|Added in Mind 1.0|
|tritonclient[all]|-|Added in Mind 1.0|
|numba|0.61.2|Added in MindIE 2.0.RC1|
|posix_ipc|1.2.0|Added in MindIE 2.2.RC1|
|fastapi|0.115.11|Added in MindIE 2.3.0|
|uvicorn|0.34.3|Added in MindIE 2.3.0|
|pybind11|3.0.1|Added in MindIE 2.3.0|
