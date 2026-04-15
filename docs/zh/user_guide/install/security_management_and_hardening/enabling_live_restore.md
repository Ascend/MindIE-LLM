# 启用现场还原live-restore

默认情况下，当Docker守护进程终止时，它将关闭正在运行的容器。用户可以配置守护程序，以便容器在守护程序不可用时保持运行，此功能称为live-restore。

live-restore选项有助于减少由于守护进程崩溃、计划中断或升级而导致的容器停机时间。

任务运行时，如果修改了Docker的配置，则需要重新加载Docker守护进程，从而导致Docker容器重启，并且业务将会中断，存在一定的风险。这种情况下，可以启用live-restore功能，当守护进程不可用时使容器保持活动状态，有以下两种方法进行设置。

**方法一：**

将配置添加到守护进程配置，即docker-daemon.json文件中，配置信息如下所示：

```json
{
    "live-restore": true
}
```

**方法二：**

使用以下命令手动启用live-restore。

```bash
dockerd --live-restore systemd
```
