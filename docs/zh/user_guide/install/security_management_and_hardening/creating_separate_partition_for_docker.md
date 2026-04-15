# 为Docker创建单独分区

Docker安装后默认目录是“/var/lib/docker”，用于存放Docker相关的文件，包括镜像、容器等。当该目录存储已满时，Docker和主机可能无法使用。因此，建议创建一个单独的分区（逻辑卷），用来存放Docker文件。

- 新安装的设备，创建一个单独的分区，用于挂载“/var/lib/docker”目录，请参见[表1](#table1)。

    **表 1**  Docker分区 <a id="table1"></a>

|分区|说明|大小|启动标志|
|--|--|--|--|
|/boot|启动分区。|500MB|on|
|/var|软件运行所产生的数据存放分区，如日志，缓存等。|>300GB|off|
|/var/lib/docker|Docker镜像与容器存放分区。<br>Docker镜像和容器默认放在“/var/lib/docker”分区下，如果“/var/lib/docker”分区使用率大于85%，K8s会启动资源驱逐机制，使用时请确保“/var/lib/docker”分区使用率在85%以下。|>300GB|off|
|/|主分区。|>300GB|off|

- 已完成安装的系统，请使用逻辑卷管理器（LVM）创建分区。
