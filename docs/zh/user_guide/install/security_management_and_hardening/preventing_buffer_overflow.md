# 缓冲区溢出安全保护

为阻止缓冲区溢出攻击，建议使用ASLR技术，通过对堆、栈、共享库映射等线性区布局的随机化，增加攻击者预测目的地址的难度，防止攻击者直接定位攻击代码位置。该技术可作用于堆、栈、内存映射区（mmap基址、shared libraries、vdso页）。

1. 确保当前用户拥有“/proc/sys/kernel/randomize\_va\_space“文件的写权限。
2. 开启缓冲区溢出安全保护。

    ```bash
    echo 2 >/proc/sys/kernel/randomize_va_space
    ```
