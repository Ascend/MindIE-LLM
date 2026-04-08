1、替换connector
cp ./connector_mock.py /usr/local/bin/connector

2、启动录制
export CONNECTOR_MOCK=0

3、启动daemon程序录制
./bin/mindieservice_daemon
curl 127.0.0.1:8977 -X POST -d '{"inputs":"Please introduce yourself."}`

4、回放
export CONNECTOR_MOCK=1

