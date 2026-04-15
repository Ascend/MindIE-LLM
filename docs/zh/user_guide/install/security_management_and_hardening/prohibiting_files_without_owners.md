# 系统中不允许存在无属主的文件

系统中不允许存在无属主的文件，如发现无属主的文件，建议删除或者改变文件的权限。

使用以下方式查找无属主文件。

```bash
find / -nouser -print
find / -nogroup -print
```
