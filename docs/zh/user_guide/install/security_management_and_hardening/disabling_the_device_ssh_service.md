# 关闭Device SSH服务

Device的SSH服务默认处于关闭状态，处于关闭状态可提升系统安全性。若已开启SSH服务，可参考以下步骤关闭。

1. 创建close\_device\_ssh.c，内容如下：

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>
    #include "dsmi_common_interface.h" // DSMI相关接口所在头文件，存储路径为Host侧的/usr/local/Ascend/driver/include。该文件也可能存储在其他路径。
    int main()
    {
           int ret;
           int dev_list[64] = {0};
           int dev_cnt = 0;
           const char config_name[20] = "ssh_status";
           unsigned int buf_size = 1;
           unsigned char buf = 0; // 0: disable 1:enable
           ret = dsmi_get_device_count(&dev_cnt);
           if (ret != 0) {
                  printf("[%s] get dev_cnt test_fail value = %d \n", __func__, ret);
                  return -1;
           }
           if (dev_cnt <= 0) {
                  printf("[%s] get dev_cnt test_fail value = -1 , dev_cnt:%d \n", __func__, dev_cnt);
                  return -1;
           }
           printf("[%s] dev_cnt:%d \n", __func__, dev_cnt);
           ret = dsmi_list_device(dev_list, dev_cnt);
           if (ret != 0) {
                  printf("[%s] list device test_fail value = %d \n", __func__, ret);
                  return -1;
           }
           for (int i = 0; i < dev_cnt; i++) {
                  ret = dsmi_set_user_config(dev_list[i], config_name, buf_size, &buf);
                  if (ret != 0) {
                         printf("[%s, %d] dev_id:%d test_fail, value = -1 ret:%d \n", __func__, __LINE__, dev_list[i], ret);
                         return -1;
                  }
                  printf("[%s, %d] dev_id:%d set %s:0x%x, buf_size:%d\n", __func__, __LINE__, dev_list[i], config_name,
                  buf, buf_size);
           }
           return 0;
    }
    ```

2. 执行如下命令，对“close\_device\_ssh.c”文件进行编译。

    ```bash
    gcc close_device_ssh.c /usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so -L. -I/usr/local/Ascend/driver/include -std=c99 -o close_device_ssh
    ```

    /usr/local/Ascend表示Driver的默认安装路径，请根据实际情况替换。

3. 执行可执行文件“close\_device\_ssh”，关闭Device的SSH服务。

    ```bash
    ./close_device_ssh
    ```

4. 重启Host。

    通过DSMI接口开启Device的SSH服务后，需要重启才能生效，请在Host侧执行如下命令进行重启操作。

    ```bash
    reboot
    ```

    > [!NOTE]说明
    > DSMI接口可以通过以下步骤开启Device的SSH服务：
    > 1. 登录DSMI接口，进入设备管理页面。
    > 2. 选择需要开启SSH服务的设备，单击“配置”按钮。
    > 3. 在设备配置页面中，选择“SSH服务”选项卡。
    > 4. 在SSH服务选项卡中，将SSH服务的状态设置为“开启”。
    > 5. 设置SSH服务的端口号，一般默认为22。
    > 6. 设置SSH服务的用户名和密码，用于登录SSH服务。
    > 7. 单击“保存”按钮，保存配置设置。
    > 完成以上步骤后，DSMI接口会自动开启设备的SSH服务，并设置相应的用户名、密码和端口号。用户可以通过SSH客户端工具连接到设备的SSH服务，进行相关的操作和管理。
