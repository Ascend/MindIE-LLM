# 升级

用户可通过安装新版本的whl包，完成MindIE升级。本章以升级MindIE LLM为例。

执行如下命令，安装新whl包，完成升级。

```
pip install mindie_llm-{version}-{python tag}-{abi tag}-{platform tag}.whl

```

> [!NOTE]说明
> 上方以mindie_llm包为例，如升级MindIE Motor或MindIE SD，请替换为对应的whl包名。

如果是同版本升级，在如上命令中加 `--force-reinstall` 参数进行强制重新安装。 

> [!CAUTION]注意
> 重装mindie_llm过程中，安装目录（/mindie_llm）下的内容会被全部删除再重装新版本，如果配置文件、证书文件等需要保留，请先备份。