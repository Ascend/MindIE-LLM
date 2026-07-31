# Disabling the Device SSH Service

By default, the device SSH service is disabled, which can improve the system security. If the SSH service has been enabled, perform the following steps to disable it:

1. Create the `close_device_ssh.c` file with the following content:

    ```cpp
    #include <stdlib.h>
    #include <stdio.h>
    #include "dsmi_common_interface.h" // Header file of the DSMI, which is stored in /usr/local/Ascend/driver/include on the host. This file may also be stored in other paths.
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

2. Run the following command to compile the `close_device_ssh.c` file:

    ```bash
    gcc close_device_ssh.c /usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so -L. -I/usr/local/Ascend/driver/include -std=c99 -o close_device_ssh
    ```

    `/usr/local/Ascend` is the default installation path of the driver. Replace it as required.

3. Run the executable file `close_device_ssh` to disable the SSH service on the device.

    ```bash
    ./close_device_ssh
    ```

4. Reboot the host.

    After the device SSH service is disabled through the DSMI, run the following command on the host to reboot the host for the configuration to take effect:

    ```bash
    reboot
    ```

> [!NOTE]
> To enable the device SSH service through the DSMI, perform the following steps:
>
> 1. Log in to the DSMI and go to the device management page.
> 2. Select the device on which the SSH service needs to be enabled and click **Configure**.
> 3. On the device configuration page, click the **SSH Service** tab.
> 4. On the **SSH Service** tab page, enable the SSH service.
> 5. Set the port number of the SSH service, which generally defaults to `22`.
> 6. Set the user name and password for logging in to the SSH service.
> 7. Click **Save** to save the settings.
> After the preceding steps are complete, the DSMI automatically enables the device SSH service and sets the user name, password, and port number. You can use the SSH client tool to connect to the device SSH service to perform related operations and management.
