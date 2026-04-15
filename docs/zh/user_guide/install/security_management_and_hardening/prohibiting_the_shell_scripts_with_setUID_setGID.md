# 禁止使用SetUID或SetGID的shell脚本

特殊权限的脚本有可能被恶意使用，给系统带来巨大的威胁，如非必要，建议禁止使用SUID、SGID脚本。

可以执行如下命令查找系统中SUID/SGID文件，查看是否必要存在。若无必要，去掉“s”位以取消文件的SetUID或SetGID权限，或删除文件。

```bash
find / -perm -2000 -exec ls -l {} \; -exec md5sum {} \;
find / -perm -4000 -exec ls -l {} \; -exec md5sum {} \;
```
