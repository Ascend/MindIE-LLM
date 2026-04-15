# 禁止使用高危capability的可执行文件

使用以下命令查询出系统中是否存在具有常见高危CAP权限的文件，常见高危CAP权限文件如[表1](#table1)所示，若无必要建议删除。

```bash
getcap -r / 2>/dev/null
```

**表 1**  常见高危CAP权限文件  <a id="table1"></a>

|capability|说明|
|--|--|
|CAP_CHOWN|改变任意文件的属主以及所属组。|
|CAP_DAC_OVERRIDE|访问文件时，忽略DAC访问限制（可理解为可绕过属主/组的权限判断）。|
|CAP_DAC_READ_SEARCH|忽略所有对读、搜索操作的限制。|
|CAP_FOWNER|可以设置任意文件属性及扩展属性，如chmod、setxattr等。|
|CAP_IPC_OWNER|访问消息队列、信号量和共享内存时，忽略访问权限检查。|
|CAP_SYS_MODULE|插入和删除内核模块。|
|CAP_SYS_PTRACE|允许跟踪任何进程。|
|CAP_SETFCAP|允许在指定的程序上授权能力给其它程序对应的二进制文件。|
