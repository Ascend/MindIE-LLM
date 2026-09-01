# Hardening Server Security

> [!NOTE]
> The Server provides only part of flow control capabilities, which do not apply to the public network. You need to guarantee Server flow control and isolation between the public network and LAN. For example, you can use the open-source software Nginx (install it by referring to [Nginx official document](https://nginx.org/en/docs/)) for isolation.

The following describes how to configure Nginx.

1. Set the Nginx configuration file. The permission on the configuration file cannot be higher than 440. The default path is `/etc/nginx/nginx.conf`.

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
    # HTTPS server configuration
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
       ssl_certificate     ${path_of_server_crt_1}; # Server certificate path, which must be configured (permissions 400)
       ssl_certificate_key ${path_of_server_key_1}; # Server private key path, which must be configured. The private key cannot be configured in plaintext. (permissions 400)
       ssl_client_certificate ${path_of_ca_crt_1}; # Root CA certificate path, which must be configured (permissions 400)

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
        proxy_pass https://127.0.0.1:1025; # Set the value to the IP address and port number configured in the MindIE Motor configuration file.
        allow 127.0.0.1; # Set the value to the remote IP address that can be accessed.
        deny all;
        proxy_ssl_certificate     ${path_of_server_crt_2}; # Server certificate path, which needs to be configured by yourself (permission 400).
        proxy_ssl_certificate_key ${path_of_server_key_2}; # Private key path of the server, which needs to be configured by yourself (permission 400). The private key cannot be configured in plaintext.
        proxy_ssl_trusted_certificate ${path_of_ca_crt_2}; # Root CA certificate path, which needs to be configured by yourself (permission 400).
        proxy_ssl_session_reuse on;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;
        proxy_ssl_ciphers "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384";
       }
      }
    }
    ```

2. Start Nginx using the `-c` option to pass the configuration file path. `${path_of_nginx_bin}` is the path of the installed Nginx binary file. Different environments or installation methods may generate different paths.

    ```text
    ${path_of_nginx_bin} -c ${path_of_nginx_config_file} # Nginx configuration file
    ```
