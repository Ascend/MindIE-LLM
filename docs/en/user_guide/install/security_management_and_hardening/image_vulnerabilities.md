# Image Vulnerabilities

Image vulnerabilities include common vulnerabilities and exposures (CVEs) in images and malicious vulnerabilities in images uploaded by attackers.

In the Dockerfile, pay attention to the CVE vulnerabilities of the base image on which the `FROM` command depends. Images are usually obtained through the official image repository Docker Hub. According to the research on image security vulnerabilities in Docker Hub, the average number of vulnerabilities in both community images and official images is close to 200.
