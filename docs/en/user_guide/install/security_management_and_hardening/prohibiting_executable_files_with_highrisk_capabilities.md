# Prohibiting Executable Files with High-risk Capabilities

Run the following command to check whether there are files with common high-risk CAP in the system. [Table 1](#table1) lists files with common high-risk CAP. You are advised to delete them if they are not necessary.

```bash
getcap -r / 2>/dev/null
```

**Table 1** Files with common high-risk CAP <a id="table1"></a>

|capability|Description|
|--|--|
|CAP_CHOWN|Changes the owner and owner group of any file.|
|CAP_DAC_OVERRIDE|Ignores DAC access restrictions during file access (i.e. bypasses the owner/group permission check).|
|CAP_DAC_READ_SEARCH|Ignores all restrictions on read and search operations.|
|CAP_FOWNER|Sets any file attributes and extended attributes, such as `chmod` and `setxattr`.|
|CAP_IPC_OWNER|Ignores access permission check when the message queue, semaphore, and shared memory are accessed.|
|CAP_SYS_MODULE|Inserts and deletes the kernel module.|
|CAP_SYS_PTRACE|Allows tracing any process.|
|CAP_SETFCAP|Allows a specified program to authorize capabilities to binary files corresponding to other programs.|
