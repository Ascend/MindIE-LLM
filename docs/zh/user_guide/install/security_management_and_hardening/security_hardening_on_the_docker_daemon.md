# Docker守护进程安全加固

- 限制容器间的相互网络访问。Docker守护进程默认允许容器间相互网络通信，容易造成信息泄露，建议将--icc=false配置在Docker守护进程中。
- 防止开启守护进程的远程管理接口。不要使用Docker Remote API服务，且需要严格控制docker.sock文件的读写权限，只将必要的用户配置到Docker用户组中。若业务需要开启Docker Remote API服务，建议通过--authorization-plugin开启守护进程的细粒度访问策略控制。
- 限制容器的文件句柄数、fork进程数。建议将--default-ulimit中nofile和nproc参数配置在Docker守护进程中，避免fork炸弹或者耗尽文件句柄资源的情况，导致主机受到攻击。容器资源限制的数值需要根据业务评估，不合理的限额会导致容器无法运行。例如--default-ulimit nofile=64:64 --default-ulimit nproc=512:512，限制单个进程的文件句柄数为64，单个UID用户的fork进程数为512。
- 禁用用户空间代理。建议将--userland-proxy=false配置在Docker守护进程中，减小攻击面。
- 启用user namespace命名空间。启用后，会提供容器用户与主机用户的权限隔离，例如使用--userns-remap=default配置在Docker守护进程中。
- 避免使用aufs存储驱动，aufs是不受支持的驱动程序。
- 配置日志驱动。根据业务评估是否需要开启日志驱动。
