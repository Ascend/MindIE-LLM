# 本地构建 MkDocs 文档服务

本文档指导开发者如何在本地构建 MkDocs 文档服务，用于文档编写时的实时预览与调试。

## 安装依赖

```shell
pip install -r requirements/mkdocs.txt
```

## 启动本地服务

Step1: 在项目根目录执行以下命令启动 MkDocs 本地服务：

```shell
mkdocs serve
```

Step2: 启动成功后，终端将输出类似以下信息：

```text
INFO     -  Building documentation...
INFO     -  Cleaning site directory
INFO     -  Documentation built in 1.23 s
INFO     -  [12:00:00] Watching paths for changes
INFO     -  [12:00:00] Serving on http://127.0.0.1:8000/
```

Step3: 在浏览器中访问 `http://127.0.0.1:8000/` 即可预览文档。

> [!NOTE]
> `mkdocs serve` 默认监听 `8000` 端口。若该端口被占用，可通过 `-a` 参数指定其他端口，例如 `mkdocs serve -a 127.0.0.1:8080`。

## 常见问题

### 依赖安装失败

若 `pip install` 过程中出现依赖冲突或安装失败，建议使用虚拟环境隔离：

```shell
python -m venv .venv_mkdocs
source .venv_mkdocs/bin/activate
pip install -r requirements/mkdocs.txt
```
