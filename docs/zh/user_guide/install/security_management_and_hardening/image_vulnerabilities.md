# 镜像漏洞

镜像漏洞安全风险主要包括镜像中的软件含有CVE（Common Vulnerabilities and Exposures）漏洞、攻击者上传含有恶意漏洞的镜像等情况。

Dockerfile文件中，FROM命令基于的基础镜像，需要用户注意基础镜像CVE漏洞。镜像的获取通常是通过官方镜像仓库Docker Hub，根据对Docker Hub中镜像安全漏洞的相关研究，无论是社区镜像还是官方镜像，其平均漏洞数均接近200个。
