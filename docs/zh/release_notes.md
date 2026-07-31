# 版本说明书

## 版本配套说明<a name="ZH-CN_TOPIC_0000002532737933"></a>

### 产品版本信息<a name="ZH-CN_TOPIC_0000002532737927"></a>

<a name="table1657153819263"></a>
<table><tbody><tr id="row35711238132618"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.1.1"><p id="p4361851192614"><a name="p4361851192614"></a><a name="p4361851192614"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.1.1 "><p id="p93605162611"><a name="p93605162611"></a><a name="p93605162611"></a>MindIE LLM</p>
</td>
</tr>
<tr id="row557118387264"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.2.1"><p id="p636175192620"><a name="p636175192620"></a><a name="p636175192620"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.2.1 "><p id="p33695111267"><a name="p33695111267"></a><a name="p33695111267"></a>3.1.0</p>
</td>
</tr>
<tr id="row3572133822619"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.3.1"><p id="p1936155113262"><a name="p1936155113262"></a><a name="p1936155113262"></a>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.3.1 "><p id="p1836125122617"><a name="p1836125122617"></a><a name="p1836125122617"></a>正式版本</p>
</td>
</tr>
<tr id="row17572183882619"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.4.1"><p id="p1636051182612"><a name="p1636051182612"></a><a name="p1636051182612"></a>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.4.1 "><p id="p4361851132614"><a name="p4361851132614"></a><a name="p4361851132614"></a>三个月</p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明<a name="ZH-CN_TOPIC_0000002500898042"></a>

| 产品名称 |  版本 |
| ---     | ---  |
| CANN    | 9.0.1 |
| MindCluster | 26.0.0 |
| Ascend Extension for PyTorch | 26.0.0 |
| CCAE    | iMaster CCAE V100R026C10SPC100 |

## 版本兼容性说明<a name="ZH-CN_TOPIC_0000002532657965"></a>

各组件需要配套使用，请勿跨版本混用。

**表 1**  软件版本兼容性说明

| MindIE LLM 版本 | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1  | CANN 8.5.0 |
| ----           | ---        | ---        | ---        | ---         | ---        |
| 3.1.0          | Y          | Y          | Y          |  /          |  /         |
| 3.0.0          | /          | /          | Y          |  Y          |  Y         |
| 2.3.0          | /          | /          | /          |  /          |  Y         |

| MindIE LLM 版本 | MindCluster 26.1.0 | MindCluster 26.0.0 | MindCluster 7.3.0 |
| ---            | ---                | ---                | ---      |
| 3.1.0          | Y                  | Y                  | Y                   |
| 3.0.0          | /                  | Y                  | Y                   |
| 2.3.0          | /                  | /                  | Y                   |

| MindIE LLM 版本 | iMaster CCAE V100R026C10SPC100 | iMaster CCAE V100R026C00SPC010 | iMaster CCAE V100R025C30SPC100 |
| ---             | ---               | ---                | --- |
| 3.1.0          | Y                  | Y                  | Y                   |
| 3.0.0          | /                  | Y                  | Y                   |
| 2.3.0          | /                  | /                  | Y                   |

## 版本使用注意事项<a name="ZH-CN_TOPIC_0000002501057896"></a>

无

## v3.1.0 更新说明<a name="ZH-CN_TOPIC_0000002532737925"></a>

MindIE LLM 组件将暂停后续功能演进，现有功能维持维护状态，不再新增特性。

### 新增特性<a name="ZH-CN_TOPIC_0000002532737923"></a>

<a name="zh-cn_topic_0000002501057442_table1287mcpsimp"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="7.1%"><p>编号</p>
</th>
<th class="cellrowborder" valign="top" width="15.129999999999999%"><p>特性</p>
</th>
<th class="cellrowborder" valign="top" width="77.77%"><p>具体内容</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="7.1%"><p>1</p>
</td>
<td class="cellrowborder" valign="top" width="15.129999999999999%"><p>功能</p>
</td>
  <td class="cellrowborder" valign="top" width="77.77%"><ul><li>基于AclGraph，支持DeepSeek-V3.2 基本功能及性能优化。</li><li>Qwen3系列模型支持LoRA+int8量化叠加。</li></ul>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>2</p>
</td>
<td class="cellrowborder" valign="top" width="15.129999999999999%"><p>性能提升</p>
</td>
<td class="cellrowborder" valign="top" width="77.77%"><ul><li>DeepSeek-V3.2 Decode 阶段使用 Triton RoPE 算子（rope_forward_triton_siso）替代原有 npu_rotary_mul，降低 RoPE 计算开销。</li><li>采样 Top-K/Top-P 使用自定义 NPU 算子实现，替代 PyTorch 软件 softmax+sort 实现，降低 TPOT。</li><li>PluginManager 新增异步 D2H（Device-to-Host）拷贝机制，采样输出通过独立 NPU Stream 异步搬移到 Host，减少推理主路径阻塞。</li></ul>
</td>
</tr>
</tbody></table>

### 修改特性<a name="ZH-CN_TOPIC_0000002500898048"></a>

- PD 混部场景新增约束：Prefix Cache、SplitFuse 与 DP（数据并行，dp>1）不可同时开启，PD 分离场景不受此限制。

### 删除特性<a name="ZH-CN_TOPIC_0000002501057899"></a>

无

### 日落特性

继承自[3.0.0版本的日落特性](https://gitcode.com/Ascend/MindIE-LLM/blob/v3.0.0/docs/zh/release_notes.md#%E6%97%A5%E8%90%BD%E7%89%B9%E6%80%A7)。

> [!NOTE]说明：以下特性将于2027年3月份日落。

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="7.1%"><p>编号</p>
</th>
<th class="cellrowborder" valign="top" width="15.67%"><p>特性</p>
</th>
<th class="cellrowborder" valign="top" width="77.23%"><p>详细</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="7.1%"><p>1</p>
</td>
<td class="cellrowborder" valign="top" width="15.67%"><p>部署形态</p>
</td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>MindIE LLM将不再支持基于run包的部署方式。Torch 2.1.0版本将随run包日落。abi0软件包将随run包日落。</li></ul>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>2</p>
</td>
<td class="cellrowborder" valign="top" width="15.67%"><p>量化</p>
</td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>W4A16量化特性。</li></ul>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>3</p>
</td>
<td class="cellrowborder" valign="top" width="15.67%"><p>测试脚本</p>
</td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>ModelTest和Benchmark工具将日落，归一至AISBench工具。</li></ul>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>4</p>
</td>
<td class="cellrowborder" valign="top" width="15.67%"><p>模型</p>
</td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>以下模型将日落：MiniCPM-1B、MiniCPM-2B、MiniCPM3-4B、Starcoder2、StableLM、yizhao、chatglm2-6b、chatglm3-6b、chatglm3-6b-32K、Qwen系列、Qwen1.5系列、Qwen2系列、Qwen2 Coder、Hunyuan、Skywork、DBRX、grok-1、Llama3.2、llava-1.5、llava-1.6、Yi-VL、VITA-1.5。</li></ul>
</td>
</tr>
</tbody>
</table>

### 接口变更说明<a name="ZH-CN_TOPIC_0000002501057888"></a>

<a name="zh-cn_topic_0000002501057442_table_interface"></a>
<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="10%"><p>变更类型</p>
</th>
<th class="cellrowborder" valign="top" width="20%"><p>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="70%"><p>变更说明</p>
</th>
</tr>
</thead>
<tbody>
<tr><td class="cellrowborder" valign="top" width="10%"><p>修改</p>
</td>
<td class="cellrowborder" valign="top" width="20%"><p>backendType支持传入"torch"字段</p>
</td>
<td class="cellrowborder" valign="top" width="70%"><p>config.json中的backendType新增支持"torch"字段，用于指定使用AclGraph后端。</p>
</td>
</tr>
</tbody></table>

### 已解决的问题<a name="ZH-CN_TOPIC_0000002501057892"></a>

无

### 遗留问题<a name="ZH-CN_TOPIC_0000002532737931"></a>

无

## 升级影响<a name="ZH-CN_TOPIC_0000002501057894"></a>

### 升级过程中对现行系统的影响<a name="ZH-CN_TOPIC_0000002532657959"></a>

- 对业务的影响
  软件版本升级过程中会导致业务中断。

- 对网络通信的影响
  对网络通信无影响。

### 升级后对现行系统的影响<a name="ZH-CN_TOPIC_0000002532657963"></a>

无

## 漏洞修补列表<a name="ZH-CN_TOPIC_0000002500898046"></a>

|软件名称|软件版本|CVE编号|实际CVSS得分|漏洞描述|解决版本|
|-------|---------|-------|----------|----------|--------|
| Transformers | unspecified 至 <4.36 | CVE-2023-6730 | 0 | Deserialization of Untrusted Data in huggingface/transformers prior to 4.36. The vulnerability exists in the RagRetriever.from_pretrained method, which allows remote attackers to execute arbitrary code via a crafted pickle file during model loading. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.36 | CVE-2023-7018 | 0 | Deserialization of Untrusted Data in huggingface/transformers prior to 4.36. The vulnerability exists in the automatic loading of vocab.pkl files from remote repositories without restrictions, allowing attackers to load malicious files and achieve remote code execution. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.48.0 | CVE-2024-12720 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability was identified in huggingface/transformers, specifically in tokenization_nougat_fast.py. The post_process_single() function uses a regex exhibiting exponential time complexity under certain conditions, leading to excessive backtracking and potential application downtime. | MindIE 3.1.0 |
| Requests | < 2.32.4 | CVE-2024-47081 | 0 | Requests is a HTTP library. Due to a URL parsing issue, Requests releases prior to 2.32.4 may leak .netrc credentials to third parties for specific maliciously-crafted URLs. | MindIE 3.1.0 |
| CPython tarfile | 0 至 \<3.8.20；3.9.0 至 \<3.9.20；3.10.0 至 \<3.10.15；3.11.0 至 \<3.11.10；3.12.0 至 \<3.12.6；3.13.0a1 至 \<3.13.0rc2 | CVE-2024-6232 | 0 | Regular expressions that allowed excessive backtracking during tarfile.TarFile header parsing are vulnerable to ReDoS via specifically-crafted tar archives. | MindIE 3.1.0 |
| Setuptools | 69.1.1 至 <70.0 | CVE-2024-6345 | 0 | A vulnerability in the package_index module of pypa/setuptools versions up to 69.1.1 allows for remote code execution via its download functions. These functions are susceptible to code injection when exposed to user-controlled inputs such as package URLs. | MindIE 3.1.0 |
| CPython email | 0 至 <=3.13.0rc2 | CVE-2024-6923 | 0 | The email module didn't properly quote newlines for email headers when serializing an email message, allowing for header injection when an email is serialized. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.50.0 | CVE-2025-1194 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause excessive CPU consumption via specially crafted input to regex patterns in the tokenization module. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14926 | 0 | Hugging Face Transformers SEW convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14927 | 0 | Hugging Face Transformers SEW-D convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14928 | 0 | Hugging Face Transformers HuBERT convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.50.0 | CVE-2025-2099 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| PyTorch | < 2.6.0 | CVE-2025-32434 | 0 | In PyTorch version 2.5.1 and prior, a Remote Command Execution (RCE) vulnerability exists when loading a model using torch.load with weights_only=True. Due to improper handling of tar format model loading, pickle deserialization can still execute arbitrary code even when the safety parameter is set. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.51.0 | CVE-2025-3262 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause excessive CPU consumption via specially crafted input. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.51.0 | CVE-2025-3263 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers configuration_utils. The get_configuration_file() function uses a vulnerable regex pattern config\.(.*)\.json that is susceptible to catastrophic backtracking, allowing attackers to cause denial of service. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.51.0 | CVE-2025-3264 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers dynamic_module_utils. The get_imports() function uses a vulnerable regex pattern to filter try/except blocks that is susceptible to catastrophic backtracking, allowing attackers to cause denial of service via crafted input. | MindIE 3.1.0 |
| PyTorch | 2.6.0 | CVE-2025-3730 | 0 | A denial of service vulnerability in PyTorch ctc_loss function allows attackers to cause excessive resource consumption via specially crafted input. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.52.1 | CVE-2025-3777 | 0 | A security vulnerability in huggingface/transformers that could be exploited to compromise the integrity or availability of the application. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.52.1 | CVE-2025-3933 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Python Protobuf | 0 至 <4.25.8；0 至 <5.29.5；0 至 <6.31.1 | CVE-2025-4565 | 0 | A denial of service vulnerability in Python Protobuf pure-Python backend. The parsing of protobuf messages can consume excessive resources, leading to denial of service. | MindIE 3.1.0 |
| Setuptools | < 78.1.1 | CVE-2025-47273 | 0 | A path traversal vulnerability in pypa/setuptools versions prior to 78.1.1 allows attackers to write arbitrary files via specially crafted package URLs. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.53.0 | CVE-2025-5197 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.53.0 | CVE-2025-6051 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.53.0 | CVE-2025-6638 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified 至 <4.53.0 | CVE-2025-6921 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| SentencePiece | All versions prior to 0.2.1 | CVE-2026-1260 | 0 | A memory safety vulnerability in SentencePiece versions prior to 0.2.1 that could allow attackers to cause denial of service or potentially execute arbitrary code. | MindIE 3.1.0 |
| Transformers | unspecified 至 <v5.0.0rc3 | CVE-2026-1839 | 0 | A deserialization of untrusted data vulnerability in huggingface/transformers prior to v5.0.0rc3 allows remote attackers to execute arbitrary code via crafted model files during loading. | MindIE 3.1.0 |

注：实际CVSS得分为0，即产品无实际漏洞攻击场景，不受漏洞影响（代码未编译、代码无调用、编译选项保护等）。
