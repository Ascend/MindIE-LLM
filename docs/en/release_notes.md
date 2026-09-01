# Release Notes

<!-- md-trans-meta sourceCommit=8f7656df9ba3aaa7896222e69f8c8f1b1ed9dd9d translatedAt=2026-08-24T08:29:21.081Z pushedAt=2026-08-24T08:29:50.855Z -->

## Version Description<a name="ZH-CN_TOPIC_0000002532737933"></a>

### Product Version Information<a name="ZH-CN_TOPIC_0000002532737927"></a>

<a name="table1657153819263"></a>

<table><tbody><tr id="row35711238132618"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.1.1"><p id="p4361851192614"><a name="p4361851192614"></a><a name="p4361851192614"></a>Product Name</p></th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.1.1 "><p id="p93605162611"><a name="p93605162611"></a><a name="p93605162611"></a>MindIE LLM</p></td>
</tr>
<tr id="row557118387264"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.2.1"><p id="p636175192620"><a name="p636175192620"></a><a name="p636175192620"></a>Product Version</p></th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.2.1 "><p id="p33695111267"><a name="p33695111267"></a><a name="p33695111267"></a>3.1.0</p></td>
</tr>
<tr id="row3572133822619"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.3.1"><p id="p1936155113262"><a name="p1936155113262"></a><a name="p1936155113262"></a>Version Type</p></th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.3.1 "><p id="p1836125122617"><a name="p1836125122617"></a><a name="p1836125122617"></a>Official</p></td>
</tr>
<tr id="row17572183882619"><th class="firstcol" valign="top" width="22.6%" id="mcps1.1.3.4.1"><p id="p1636051182612"><a name="p1636051182612"></a><a name="p1636051182612"></a>Maintenance Period</p></th>
<td class="cellrowborder" valign="top" width="77.4%" headers="mcps1.1.3.4.1 "><p id="p4361851132614"><a name="p4361851132614"></a><a name="p4361851132614"></a>Three months</p></td>
</tr>
</tbody>
</table>

### Version Mapping of Related Products<a name="ZH-CN_TOPIC_0000002500898042"></a>

| Product Name | Version |
| ---     | ---  |
| CANN    | 9.0.1 |
| MindCluster | 26.0.0 |
| TorchNPU | 26.0.0 |
| CCAE    | iMaster CCAE V100R026C10SPC100 |

## Version Compatibility<a name="ZH-CN_TOPIC_0000002532657965"></a>

All components must be used together as a matched set. Do not mix versions across releases.

**Table 1** Software version compatibility

| MindIE LLM Version | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1  | CANN 8.5.0 |
| ----           | ---        | ---        | ---        | ---         | ---        |
| 3.1.0          | Y          | Y          | Y          |  /          |  /         |
| 3.0.0          | /          | /          | Y          |  Y          |  Y         |
| 2.3.0          | /          | /          | /          |  /          |  Y         |

| MindIE LLM Version | MindCluster 26.1.0 | MindCluster 26.0.0 | MindCluster 7.3.0 |
| ---            | ---                | ---                | ---      |
| 3.1.0          | Y                  | Y                  | Y                   |
| 3.0.0          | /                  | Y                  | Y                   |
| 2.3.0          | /                  | /                  | Y                   |

| MindIE LLM Version | iMaster CCAE V100R026C10SPC100 | iMaster CCAE V100R026C00SPC010 | iMaster CCAE V100R025C30SPC100 |
| ---             | ---               | ---                | --- |
| 3.1.0          | Y                  | Y                  | Y                   |
| 3.0.0          | /                  | Y                  | Y                   |
| 2.3.0          | /                  | /                  | Y                   |

## Version Usage Notes<a name="ZH-CN_TOPIC_0000002501057896"></a>

None

## v3.1.0 Update Notes<a name="ZH-CN_TOPIC_0000002532737925"></a>

The MindIE LLM component will suspend further feature evolution. Existing features will remain in maintenance status, and new features and new models will no longer be supported. It is recommended to use MindIE Motor + vLLM Ascend to deploy inference services. For quick deployment of vLLM Ascend, see [Quick Start](https://docs.vllm.ai/projects/ascend/en/v0.23.0/quick_start.html).

### New Features<a name="ZH-CN_TOPIC_0000002532737923"></a>

<a name="zh-cn_topic_0000002501057442_table1287mcpsimp"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="7.1%"><p>Number</p></th>
<th class="cellrowborder" valign="top" width="15.129999999999999%"><p>Feature</p></th>
<th class="cellrowborder" valign="top" width="77.77%"><p>Details</p></th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="7.1%"><p>1</p></td>
<td class="cellrowborder" valign="top" width="15.129999999999999%"><p>Function</p></td>
  <td class="cellrowborder" valign="top" width="77.77%"><ul><li>Based on AclGraph, supports the basic functions and performance optimization of DeepSeek-V3.2.</li><li>Qwen3 series models support the combination of LoRA and int8 quantization.</li></ul></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>2</p></td>
<td class="cellrowborder" valign="top" width="15.129999999999999%"><p>Performance improvement</p></td>
<td class="cellrowborder" valign="top" width="77.77%"><ul><li>In the DeepSeek-V3.2 Decode phase, the Triton RoPE operator (rope_forward_triton_siso) is used to replace the original npu_rotary_mul, reducing the RoPE computation overhead.</li><li>Top-K/Top-P sampling is implemented using custom NPU operators, replacing the PyTorch software softmax+sort implementation to reduce TPOT.</li><li>PluginManager adds an asynchronous D2H (Device-to-Host) copy mechanism. Sampling output is asynchronously moved to the Host through an independent NPU Stream, reducing blocking on the main inference path.</li></ul></td>
</tr>
</tbody></table>

### Modified Features<a name="ZH-CN_TOPIC_0000002500898048"></a>

- New constraint added for the PD co-location scenario: Prefix Cache, SplitFuse, and DP (data parallelism, dp>1) cannot be enabled simultaneously. The PD disaggregation scenario is not subject to this constraint.

### Deleted Features<a name="ZH-CN_TOPIC_0000002501057899"></a>

None

### Deprecated Features

Inherited from the [deprecated features of version 3.0.0](https://gitcode.com/Ascend/MindIE-LLM/blob/v3.0.0/docs/zh/release_notes.md#%E6%97%A5%E8%90%BD%E7%89%B9%E6%80%A7).

> [!NOTE]Description: The following features will be deprecated in March 2027.

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="7.1%"><p>Number</p></th>
<th class="cellrowborder" valign="top" width="15.67%"><p>Feature</p></th>
<th class="cellrowborder" valign="top" width="77.23%"><p>Details</p></th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="7.1%"><p>1</p></td>
<td class="cellrowborder" valign="top" width="15.67%"><p>Deployment Form</p></td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>MindIE LLM will no longer support the run package-based deployment method. Torch 2.1.0 will be sunset along with the run package. The abi0 software package will be sunset along with the run package.</li></ul></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>2</p></td>
<td class="cellrowborder" valign="top" width="15.67%"><p>Quantization</p></td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>W4A16 quantization feature.</li></ul></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>3</p></td>
<td class="cellrowborder" valign="top" width="15.67%"><p>Test script</p></td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>The ModelTest and Benchmark tools will be sunset and unified into the AISBench tool.</li></ul></td>
</tr>
<tr><td class="cellrowborder" valign="top" width="7.1%"><p>4</p></td>
<td class="cellrowborder" valign="top" width="15.67%"><p>Model</p></td>
<td class="cellrowborder" valign="top" width="77.23%"><ul><li>The following models will be sunset: MiniCPM-1B, MiniCPM-2B, MiniCPM3-4B, Starcoder2, StableLM, yizhao, chatglm2-6b, chatglm3-6b, chatglm3-6b-32K, Qwen series, Qwen1.5 series, Qwen2 series, Qwen2 Coder, Hunyuan, Skywork, DBRX, grok-1, Llama3.2, llava-1.5, llava-1.6, Yi-VL, and VITA-1.5.</li></ul></td>
</tr>
</tbody>
</table>

### Interface Change Description<a name="ZH-CN_TOPIC_0000002501057888"></a>

<a name="zh-cn_topic_0000002501057442_table_interface"></a>

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="10%"><p>Change Type</p></th>
<th class="cellrowborder" valign="top" width="20%"><p>Interface Name</p></th>
<th class="cellrowborder" valign="top" width="70%"><p>Change Description</p></th>
</tr>
</thead>
<tbody>
<tr><td class="cellrowborder" valign="top" width="10%"><p>Modified</p></td>
<td class="cellrowborder" valign="top" width="20%"><p>backendType supports the "torch" field</p></td>
<td class="cellrowborder" valign="top" width="70%"><p>The backendType in config.json now supports the "torch" field, which is used to specify the use of the AclGraph backend.</p></td>
</tr>
</tbody></table>

### Resolved Issues<a name="ZH-CN_TOPIC_0000002501057892"></a>

None

### Known Issues<a name="ZH-CN_TOPIC_0000002532737931"></a>

None

## Upgrade Impact<a name="ZH-CN_TOPIC_0000002501057894"></a>

### Impact on the Current System During the Upgrade<a name="ZH-CN_TOPIC_0000002532657959"></a>

- Impact on services

  Service interruption occurs during the software version upgrade.

- Impact on network communication

  There is no impact on network communication.

### Impact on the Current System After Upgrade<a name="ZH-CN_TOPIC_0000002532657963"></a>

None

## Vulnerability Patch List<a name="ZH-CN_TOPIC_0000002500898046"></a>

|Software Name|Software Version|CVE Number|Actual CVSS Score|Vulnerability Description|Resolved Version|
|-------|---------|-------|----------|----------|--------|
| Transformers | unspecified to <4.36 | CVE-2023-6730 | 0 | Deserialization of Untrusted Data in huggingface/transformers prior to 4.36. The vulnerability exists in the RagRetriever.from_pretrained method, which allows remote attackers to execute arbitrary code via a crafted pickle file during model loading. | MindIE 3.1.0 |
| Transformers | unspecified to <4.36 | CVE-2023-7018 | 0 | Deserialization of Untrusted Data in huggingface/transformers prior to 4.36. The vulnerability exists in the automatic loading of vocab.pkl files from remote repositories without restrictions, allowing attackers to load malicious files and achieve remote code execution. | MindIE 3.1.0 |
| Transformers | unspecified to <4.48.0 | CVE-2024-12720 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability was identified in huggingface/transformers, specifically in tokenization_nougat_fast.py. The post_process_single() function uses a regex exhibiting exponential time complexity under certain conditions, leading to excessive backtracking and potential application downtime. | MindIE 3.1.0 |
| Requests | < 2.32.4 | CVE-2024-47081 | 0 | Requests is a HTTP library. Due to a URL parsing issue, Requests releases prior to 2.32.4 may leak .netrc credentials to third parties for specific maliciously-crafted URLs. | MindIE 3.1.0 |
| CPython tarfile | 0 to \<3.8.20; 3.9.0 to \<3.9.20; 3.10.0 to \<3.10.15; 3.11.0 to \<3.11.10; 3.12.0 to \<3.12.6; 3.13.0a1 to \<3.13.0rc2 | CVE-2024-6232 | 0 | Regular expressions that allowed excessive backtracking during tarfile.TarFile header parsing are vulnerable to ReDoS via specifically-crafted tar archives. | MindIE 3.1.0 |
| Setuptools | 69.1.1 to <70.0 | CVE-2024-6345 | 0 | A vulnerability in the package_index module of pypa/setuptools versions up to 69.1.1 allows for remote code execution via its download functions. These functions are susceptible to code injection when exposed to user-controlled inputs such as package URLs. | MindIE 3.1.0 |
| CPython email | 0 to <=3.13.0rc2 | CVE-2024-6923 | 0 | The email module didn't properly quote newlines for email headers when serializing an email message, allowing for header injection when an email is serialized. | MindIE 3.1.0 |
| Transformers | unspecified to <4.50.0 | CVE-2025-1194 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause excessive CPU consumption via specially crafted input to regex patterns in the tokenization module. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14926 | 0 | Hugging Face Transformers SEW convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14927 | 0 | Hugging Face Transformers SEW-D convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | 4.57.0 | CVE-2025-14928 | 0 | Hugging Face Transformers HuBERT convert_config Code Injection Remote Code Execution Vulnerability. This vulnerability allows remote attackers to execute arbitrary code on affected installations. The specific flaw exists within the convert_config function, where lack of proper validation of a user-supplied string allows execution of Python code. | MindIE 3.1.0 |
| Transformers | unspecified to <4.50.0 | CVE-2025-2099 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| PyTorch | < 2.6.0 | CVE-2025-32434 | 0 | In PyTorch version 2.5.1 and prior, a Remote Command Execution (RCE) vulnerability exists when loading a model using torch.load with weights_only=True. Due to improper handling of tar format model loading, pickle deserialization can still execute arbitrary code even when the safety parameter is set. | MindIE 3.1.0 |
| Transformers | unspecified to <4.51.0 | CVE-2025-3262 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause excessive CPU consumption via specially crafted input. | MindIE 3.1.0 |
| Transformers | unspecified to <4.51.0 | CVE-2025-3263 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers configuration_utils. The get_configuration_file() function uses a vulnerable regex pattern config\.(.*)\.json that is susceptible to catastrophic backtracking, allowing attackers to cause denial of service. | MindIE 3.1.0 |
| Transformers | unspecified to <4.51.0 | CVE-2025-3264 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers dynamic_module_utils. The get_imports() function uses a vulnerable regex pattern to filter try/except blocks that is susceptible to catastrophic backtracking, allowing attackers to cause denial of service via crafted input. | MindIE 3.1.0 |
| PyTorch | 2.6.0 | CVE-2025-3730 | 0 | A denial of service vulnerability in PyTorch ctc_loss function allows attackers to cause excessive resource consumption via specially crafted input. | MindIE 3.1.0 |
| Transformers | unspecified to <4.52.1 | CVE-2025-3777 | 0 | A security vulnerability in huggingface/transformers that could be exploited to compromise the integrity or availability of the application. | MindIE 3.1.0 |
| Transformers | unspecified to <4.52.1 | CVE-2025-3933 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Python Protobuf | 0 to <4.25.8; 0 to <5.29.5; 0 to <6.31.1 | CVE-2025-4565 | 0 | A denial of service vulnerability in Python Protobuf pure-Python backend. The parsing of protobuf messages can consume excessive resources, leading to denial of service. | MindIE 3.1.0 |
| Setuptools | < 78.1.1 | CVE-2025-47273 | 0 | A path traversal vulnerability in pypa/setuptools versions prior to 78.1.1 allows attackers to write arbitrary files via specially crafted package URLs. | MindIE 3.1.0 |
| Transformers | unspecified to <4.53.0 | CVE-2025-5197 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified to <4.53.0 | CVE-2025-6051 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified to <4.53.0 | CVE-2025-6638 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| Transformers | unspecified to <4.53.0 | CVE-2025-6921 | 0 | A Regular Expression Denial of Service (ReDoS) vulnerability in huggingface/transformers allows attackers to cause denial of service via specially crafted input to regex patterns. | MindIE 3.1.0 |
| SentencePiece | All versions prior to 0.2.1 | CVE-2026-1260 | 0 | A memory safety vulnerability in SentencePiece versions prior to 0.2.1 that could allow attackers to cause denial of service or potentially execute arbitrary code. | MindIE 3.1.0 |
| Transformers | unspecified to <v5.0.0rc3 | CVE-2026-1839 | 0 | A deserialization of untrusted data vulnerability in huggingface/transformers prior to v5.0.0rc3 allows remote attackers to execute arbitrary code via crafted model files during loading. | MindIE 3.1.0 |

Note: The actual CVSS score is 0, which means the product has no actual vulnerability attack scenario and is not affected by the vulnerability (code not compiled, code not invoked, compilation option protection, etc.).
