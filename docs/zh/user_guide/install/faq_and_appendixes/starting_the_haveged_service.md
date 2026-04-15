# 启动haveged服务

## 前提条件

环境已安装haveged，如未安装，请自行安装。

## 操作步骤

> [!NOTE]说明
>
> - Server支持使用HTTPS双向认证，对客户端发起的HTTP请求进行身份认证。当开启HTTPS时，为了给服务器密钥进行口令加密，加密工具KMC使用的随机数生成算法需要haveged组件进行补熵。需要在所有Server服务安装节点上执行以下操作。
> - MindIE MS服务端使用了KMC工具，部署MS的管理节点同样需要haveged组件补熵。
> - MindIE MS部署MindIE Server时在容器内自动生成证书需要调用KMC解密，生成随机口令，对熵值（4096）要求较高，需要在计算节点安装haveged组件补熵。

请在当前Linux环境中检查是否补熵，查看和补熵的操作如下：

1. 确认系统是否开启了haveged服务（建议一直开启）。

    ```bash
    systemctl status haveged.service
    ```

    或

    ```bash
    ps -ef | grep "haveged" | grep -v "grep"
    ```

2. 修改/etc/default/haveged配置文件的熵值为4096。

    ```bash
    DAEMON_ARGS="-w 4096"
    ```

3. 启动haveged服务，并将其设置为随系统启动，确保haveged服务一直开启。

    ```bash
    systemctl start haveged.service
    systemctl enable haveged.service
    ```

4. 查看屏幕输出随机数的速度。

    ```bash
    cat /dev/random | od -x
    ```

    查看当前熵值。

    ```bash
    cat /proc/sys/kernel/random/entropy_avail
    ```

    正常情况下，未启动haveged是100多，启动haveged之后会相应增大。
