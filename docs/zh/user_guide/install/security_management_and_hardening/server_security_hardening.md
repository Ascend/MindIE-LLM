# Server安全加固

> [!NOTE]说明
> Server仅提供部分流控能力，且不直接对接公网，Server流控和公网、局域网隔离由用户保证。如可以使用开源软件Nginx进行保障，用户可参照[Nginx官方文档](http://nginx.org/en/docs/)进行Nginx的部署。

以Nginx为例进行Nginx的配置。

1. 设置Nginx配置文件，配置文件要求权限不高于440，默认路径为: /etc/nginx/nginx.conf。

    ```text
    worker_processes 1;
    worker_cpu_affinity 0001;
    worker_rlimit_nofile 4096;
    events {
        worker_connections 4096;
    }
    http {
     port_in_redirect off;
     server_tokens off;
     autoindex off;

     log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                          '$status $body_bytes_sent "$http_referer" '
                          '"$http_user_agent" "$http_x_forwarded_for" "$request_time"';

     access_log /var/log/nginx/access.log main;
     error_log /var/log/nginx/error.log info;
     limit_req_zone global zone=req_zone:100m rate=20r/s;
     limit_conn_zone global zone=north_conn_zone:100m;
    # HTTPS请求
      server {
       listen 127.0.0.1:8082 ssl;
       server_name localhost;

       add_header Referrer-Policy "no-referrer";
       add_header X-XSS-Protection "1; mode=block";
       add_header X-Frame-Options DENY;
       add_header X-Content-Type-Options nosniff;
       add_header Strict-Transport-Security " max-age=31536000; includeSubDomains ";
       add_header Content-Security-Policy "default-src 'self'";
       add_header Cache-control "no-cache, no-store, must-revalidate";
       add_header Pragma no-cache;
       add_header Expires 0;
       ssl_session_tickets off;
       ssl_certificate     ${path_of_server_crt_1}; # 服务端证书路径，需要用户自行配置(权限400)
       ssl_certificate_key ${path_of_server_key_1}; # 服务端私钥路径，需要用户自行配置，私钥不能明文配置(权限400)
       ssl_client_certificate ${path_of_ca_crt_1}; # 根ca证书路径，需要用户自行配置(权限400)

       send_timeout 60;
       limit_req zone=req_zone burst=20 nodelay;
       limit_conn north_conn_zone 20;
       keepalive_timeout  60;
       proxy_read_timeout 900;
       proxy_connect_timeout   60;
       proxy_send_timeout      60;
       client_header_timeout   60;
       client_body_timeout 10;
       client_header_buffer_size  2k;
       large_client_header_buffers 4 8k;
       client_body_buffer_size 16K;
       client_max_body_size 20m;
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256 !aNULL !eNULL !LOW !3DES !MD5 !EXP !PSK !SRP !DSS !RC4";

       ssl_verify_client on;
       ssl_verify_depth 9;
       ssl_session_timeout 10s;
       ssl_session_cache shared:SSL:10m;
       location / {
        proxy_pass https://127.0.0.1:1025; # 需要设置为MindIE Motor配置文件配置的ip及端口
        allow 127.0.0.1; #需要设置允许访问的远端ip
        deny all;
        proxy_ssl_certificate     ${path_of_server_crt_2}; # 服务端证书路径，需要用户自行配置 (权限400)
        proxy_ssl_certificate_key ${path_of_server_key_2}; # 服务端私钥路径，需要用户自行配置，私钥不能明文配置 (权限400)
        proxy_ssl_trusted_certificate ${path_of_ca_crt_2}; # 根ca证书路径，需要用户自行配置 (权限400)
        proxy_ssl_session_reuse on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;
        proxy_ssl_ciphers "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384";
       }
      }
    }
    ```

2. 启动Nginx，使用**-c**命令传入配置文件路径。$\{path\_of\_nginx\_bin\}为已安装的Nginx的二进制路径，不同环境或者安装方式生成的路径可能不同。

    ```text
    ${path_of_nginx_bin} -c ${path_of_nginx_config_file} # nginx配置文件
    ```
