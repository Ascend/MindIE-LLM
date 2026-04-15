# 防火墙设置

操作系统安装后，若配置普通用户，可以通过在“/etc/login.defs“文件中新增“ALWAYS\_SET\_PATH=yes”配置，防止越权操作。此外，为了防止使用“su”命令切换用户时，将当前用户环境变量带入其他环境造成提权。

请使用“su - \[user\]”命令进行用户切换，同时在服务器配置文件“/etc/default/su“中增加配置参数“ALWAYS\_SET\_PATH=yes”防止提权。
