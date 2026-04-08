# 免责声明！！！

## 当前仅仅提供权重加解密的一个范例，不对安全性进行负责，用户需根据具体场景使用自己的加密脚本进行替换。

# 模型加解密使用指导
当前解密支持显存预分配权重加载方式解密，传统路线权重加载方式的解密仅作为示例。
- `显存预分配权重加载方式（推荐使用）`：当前版本主要采用的加载方式。其解密支持服务化、逐Tensor加载权重等新特性。
- `传统路线权重加载方式(仅作为历史版本示例)`：历史版本默认的权重加载方式，又称“C++组图”。解密不支持逐Tensor加载权重等新特性，不支持服务化。

## 显存预分配权重加载方式加解密使用指导
### 准备工作
确认此目录`encrypttest`的路径与`${atb_models}`的相对位置与此仓库保持一致，即`${atb_models}/tests/encrypttest`。
- 如下载本仓库并编译使用，默认路径即符合，无需其他工作。
- 如使用MindIE自带镜像包，则`${atb_models}`默认路径为`/usr/local/Ascend/atb_models`，默认不包含此目录，需从网页下载此目录全部内容（`encrypttest`），并手动将目录放至符合`${atb_models}/tests/encrypttest/`

### 加密脚本介绍

用户需使用本仓的接口实现加密逻辑，接口位置为`${atb_models}/tests/encrypttest/custom_crypt.py`中的`CustomDecrypt`类：

```python
class CustomEncrypt(Encrypt):

    def __init__(self):
        super().__init__()
        self.generate_keys()

    def generate_keys(self):
        raise NotImplementedError("Please implement your method for gnerating secret keys.")

    def encrypt(self, tensor: torch.Tensor):
        """
        Implemet your encrypting method here

        Returns an encrypted tensor

        Args:
            tensor(`torch.Tensor`):
                The tensor you want to encrypt

        Returns:
            (`torch.Tensor`):
                The encrypted tensor, obtained by encrypting the input tensor

        """
        raise NotImplementedError("Please implement your method to encrypt tesnors.")
```
* 用户需要在`generate_keys`方法中自行实现安全生成秘钥的逻辑，并在`encrypt`方法中实现对传入的某个Tensor加密的逻辑。
  
* 生成的加密后权重主目录需以“crypt”作为后缀。例如加密前的权重目录为“Qwen2.5-7B-Instruct”，则加密后的权重目录应为“Qwen2.5-7B-Instruct-crypt”。
  
* 如用户对自定义加密的具体实现感到疑惑，可参考下方 `传统路线权重加载方式加密脚本介绍` 中的示例文件（`${atb_models}/tests/encrypttest/encrypt.py`的`EncryptTools`类来模拟实现自定义加密。
  
* 注意，示例文件仅为用户理解加密方法使用，请在**安全**的**离线开发环境**模拟使用。在生产环境中，用户**不可直接照搬示例的加密方式**，必须自行实现**安全的加密逻辑**。

### 解密脚本介绍

用户解密逻辑请使用本仓的接口，接口位置为`${atb_models}/tests/encrypttest/custom_crypt.py`中的`CustomEncrypt`类：
```python
class CustomDecrypt(Decrypt):

    def __init__(self):
        super().__init__()
        self.get_key_paths()

    def get_key_paths(self):
        raise NotImplementedError("Please implemet your method to get your keys.")

    def decrypt(self, encrypted_tensor: torch.Tensor):
        """
        Implemet your decrypting method here

        Returns an decrypted tensor

        Args:
            encrypted_tensor(`torch.Tensor`):
                The encrypted tensor you want to decrypt

        Returns:
            (`torch.Tensor`):
                The decrypted tensor, obtained by decrypting the input encrypted tesnor

        """
        raise NotImplementedError("Please implement your method to decrypt tesnors.")
```

* 用户需要在`get_key_paths`方法中自行实现安全获得秘钥的逻辑，并在`decrypt`方法中实现对传入的经过加密的Tensor解密的逻辑。

* 如用户对自定义解密的具体实现感到困惑，可参考下方 `传统权重加载方式解密脚本介绍` 中的示例文件（`${atb_models}/tests/decrypttest/decrypt.py`的`DecryptTools`类。示例文件仅为用户理解解密方法使用，请在**安全**的**离线开发环境**模拟使用。在生产环境中，用户**不可直接照搬示例的解密方式**，必须自行实现**安全的解密逻辑**。

### 加密阶段

使用脚本 `encrypt_weights.py` ，输入需要加密的模型权重路径和加密后的权重路径。

在用户的真实场景中，加密阶段一般在用户设备进行，所生成的密钥等信息也由用户决定设置在具体的存储位置，请使用**安全**的加密方法并**妥善保管**自己的**秘钥**。

**注意事项：**

1. 选择**安全的加密算法**，例如：AES-256, CTR模式加密算法；
2. 密钥**不要用明文**存储；
3. 对权重和密钥文件存储路径进行合法性和安全性校验。

#### 模型加密运行脚本示例

```bash
cd ${atb_models}
python tests/encrypttest/encrypt_weights.py --model_weights_path /your/model/path --encrypted_model_weights_path /your/encrypt_model/path --key_path /your/key/path
```

tips: 如果自己在上述加密脚本`custom_crypt.py`中自定义了加密方法`CustomEncrypt`，那么直接执行成功，否则需要用户输入“yes/y”来确定是否使用示例加密方法(`encrypt.py`中的`EncryptTools`)。


### 解密推理阶段

使用脚本 `decrypt_code_padding_v2.py` 为整个推理流程增加解密代码。使用方法如下：

```bash
python3 decrypt_code_padding_v2.py
```

完成上述操作之后，即可实现使用加密后的模型权重进行推理。当然，用户需实现：
1. 上述解密脚本介绍的`CustomDecrypt`类中的方法。
2. 加密后的权重后缀必须以“crypt”结尾。

#### 加密权重“纯模型”推理示例：

```bash
# 使用多卡运行Paged Attention，设置模型权重路径，设置输出长度为2048个token
cd ${atb_models}
torchrun --nproc_per_node 2 --master_port 20038 -m examples.run_pa --model_path ${encrypt_weight_path} --max_output_length 2048
```
其余参数的解释请参考 `run_pa` 的[介绍文档](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md)。

#### 加密权重“服务化”推理示例

“服务化”推理用户不感知解密过程。具体操作请参考昇腾社区MindIE服务化部分的[说明文档](https://www.hiascend.com/document/detail/zh/mindie/21RC1/mindieservice/servicedev/mindie_service0001.html)

#### 使用下游问答数据集的精度测试例子。

使用下游数据集进行测试时，先下载好数据集。

```bash
##单机场景
cd ${atb_models}/tests/modeltest
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) (lora [lora_data_path]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])

## 示例
bash run.sh pa_fp16 full_TruthfulQA 4 llama /your/model/path 8

```
参数的解释请参考modeltest下的[介绍文档](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/tests/modeltest/README.md) 中的  精度测试（下游数据集）章节。


## 传统路线权重加载方式加解密使用指导

本路线解密仅为用户理解历史版本的示例，不推荐实际使用。

### 准备工作

同显存预分配权重加载方式

### 加解密脚本介绍

#### encrypt.py

将加密相关的信息放在自定义加密类中；加密算法在类中的encrypt方法中实现

```python
class EncryptTools(Encrypt):
    def __init__(self) -> None:
        super().__init__()

        # generate key
        self.key = your key info

    def encrypt(self, tensor: torch.Tensor):
        """
        输入是原始tensor，输出是加密tensor。
        保证加密前后，encrypted_tensor和tensor的shape一致。
        """
	return encrypted_tensor
```

本仓库给出脚本中的key生成只是一种使用范例，加密算法使用的AES-256，CTR模式。

若使用本仓库的AES加密算法，请先安装 `cryptography` 。

```bash
pip install cryptography==45.0.4
```

#### decrypt.py

将解密相关的信息放在自定义解密类中；解密算法在类中的decrypt方法中实现

```python
class DecryptTools(Decrypt):
    def __init__(self) -> None:
        super().__init__()

        # gain key
        self.key = your key info

    def decrypt(self, encrypted_tensor: torch.Tensor):
        """
        输入是加密tensor，输出是解密tensor。 
        保证解密前后，encrypted_tensor和decrypted_tensor的shape一致。
        """
	return decrypted_tensor
```

本仓库给出脚本中的key只是一种使用范例，解密算法使用的AES-256，CTR模式。

### 加密阶段

使用脚本 `encrypt_weights.py` ，输入需要加密的模型权重路径和加密后的权重路径。

在用户的真实场景中，加密阶段一般在用户设备进行，所生成的密钥等信息也由用户决定设置在具体的存储位置，密钥等信息是存储在用户自己的设备上的。

**注意事项：**

1. 选择安全的加密算法，例如：AES-256, CTR模式加密算法；
2. 密钥不要用明文存储；
3. 对权重和密钥文件存储路径进行合法性和安全性校验。

#### 模型加密运行脚本示例

```bash
cd ${atb_models}
python tests/encrypttest/encrypt_weights.py --model_weights_path /your/model/path --encrypted_model_weights_path /your/encrypt_model/path --key_path /your/key/path
```

若想使用自定义加密算法，修改加密脚本 `encrypt.py` 中的加密函数。

### 解密推理阶段


使用脚本 `decrypt_code_padding.sh` 给 `${atb_models}atb_llm/utils/` 路径下的 `weights.py` 文件增加解密代码。使用方法如下：

```bash
bash decrypt_code_padding.sh ${atb_models}/atb_llm/utils/weights.py
```

完成上述操作之后，即可实现使用加密后的模型权重进行推理。同时解密脚本中含有针对权重文件的解密方法接口，用户可以传入加密权重文件路径将加密权重进行解密。当然，针对文件的具体解密算法，用户需重新实现DecryptTools类中的 `decrypt`方法。

#### 纯推理 运行脚本示例：

```bash
# 使用多卡运行Paged Attention，设置模型权重路径，设置输出长度为2048个token
cd ${atb_models}
torchrun --nproc_per_node 2 --master_port 20038 -m examples.run_pa --model_path ${encrypt_weight_path} --max_output_length 2048 --kw_args '{"encrypt_enable": 1, "key_path": "/your/key/path"}'
```

其中，

- encrypt_enable 参数控制是否可以加载加密的模型权重，默认为0。
- key_path  密钥路径。

其余参数的解释请参考 `run_pa` 的[介绍文档](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md)。

#### 服务化 运行：
暂不支持服务化运行。

#### 使用下游问答数据集的精度测试例子。

使用下游数据集进行测试时，先下载好数据集。

```bash
##单机场景
cd ${atb_models}/tests/modeltest
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_model]) (lora [lora_data_path]) [weight_dir] ([encrypt_info]) [chip_num] ([max_position_embedding/max_sequence_length])

## 示例
bash run.sh pa_fp16 full_TruthfulQA 4 llama /your/model/path '{"encrypt_enable": 1, "key_path": "/your/key/path"}' 8

```

其中，encrypt_enable 参数控制是否可以加载加密的模型权重， key_path 表示加密权重的密钥路径。

其余参数的解释请参考modeltest下的[介绍文档](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/tests/modeltest/README.md) 中的  精度测试（下游数据集）章节。
