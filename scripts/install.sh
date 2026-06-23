#!/bin/bash
set -e
sourcedir=$PWD
VERSION=VERSION_PLACEHOLDER
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER

install_flag=n
install_path_flag=n
quiet_flag=n
install_dir=""
MAX_LOG_SIZE=$((1024*1024*50))

ori_umsk=$(umask)
umask 0027 # The permission mask for creating log files(640) and directories(750).

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi

function exit_solver() {
    if [ -f "$log_file" ]; then
        chmod 440 ${log_file}
    fi
    if [ -d "$LOG_PATH" ]; then
        chmod 750 ${LOG_PATH}
    fi
    exit_code=$?
    if [ ${exit_code} -ne 0 ];then
        print "ERROR" "Install failed, [ERROR] ret code:${exit_code}"
        exit ${exit_code}
    fi
    exit 0
}

trap exit_solver EXIT

# 将日志记录到日志文件
function log() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    chmod 750 ${LOG_PATH}
    chmod 640 $log_file
    if [ x"$log_file" = x ]; then
        echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
        else
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" >>$log_file
        fi
    fi
    chmod 440 $log_file
}

# 将日志记录到日志文件并打屏
function print() {
    if [ ! -f "$log_file" ]; then
        touch $log_file
    fi
    chmod 640 $log_file
    if [ x"$log_file" = x ]; then
        echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
        else
            echo -e "[mindie-llm] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
        fi
    fi
    chmod 440 $log_file
}

# 创建文件夹
function make_dir() {
    log "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

# 创建文件
function make_file() {
    log "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

## 日志模块初始化 ##
function log_init() {
    # 判断输入的日志保存路径是否存在，不存在就创建
    if [ ! -d "$LOG_PATH" ]; then
        make_dir "$LOG_PATH"
    fi
    chmod 750 ${LOG_PATH}
    # 判断日志文件是否存在，如果不存在就创建；存在则判断是否大于50M
    if [ ! -f "$log_file" ]; then
        make_file "$log_file"
    else
        local filesize=$(ls -l $log_file | awk '{ print $5}')
        local maxsize=$((1024*1024*50))
        if [ $filesize -gt $maxsize ]; then
            local log_file_move_name="ascend_llm_install_bak.log"
            chmod 640 ${log_file}
            mv -f ${log_file} ${LOG_PATH}${log_file_move_name}
            chmod 440 ${LOG_PATH}${log_file_move_name}
            make_file "$log_file"
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}${log_file_move_name}."
        fi
        chmod 640 ${log_file}
    fi
    print "INFO" "Install log save in ${log_file}"
}

function chmod_authority() {
    # 修改文件和目录权限
    chmod_file ${default_install_path}
    chmod_file ${install_dir}
    chmod 550 ${install_dir}/scripts/uninstall.sh
    chmod_dir ${default_install_path} "550"
}

function chmod_file() {
    chmod_recursion ${1} "550" "file" "*.sh"
    chmod_recursion ${1} "440" "file" "*.bin"
    chmod_recursion ${1} "440" "file" "*.h"
    chmod_recursion ${1} "440" "file" "*.info"
    chmod_recursion ${1} "440" "file" "*.so"
    chmod_recursion ${1} "440" "file" "*.so.*"
    chmod_recursion ${1} "440" "file" "*.a"
    chmod_recursion ${1} "440" "file" "*.ini"
    chmod_recursion ${1} "550" "file" "*.py"
    chmod_recursion ${1} "550" "file" "*.whl"
    chmod_recursion ${1} "500" "file" "mindie_llm_backend_connector"
}

function chmod_dir() {
    chmod_recursion ${1} ${2} "dir"
}

function chmod_recursion() {
    local rights=$2
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -name $4 -exec chmod ${rights} {} \; 2>/dev/null
    fi
}

function parse_script_args() {
    target_dir=""
    while true
    do
        case "$1" in
        --quiet)
            quiet_flag=y
            shift
        ;;
        --install)
        install_flag=y
        shift
        ;;
        --install-path=*)
        install_path_flag=y
        target_dir=$(echo $1 | cut -d"=" -f2-)
        target_dir=${target_dir}/mindie_llm
        shift
        ;;
        --*)
        shift
        ;;
        *)
        break
        ;;
        esac
    done
}

function check_target_dir_owner() {
    local cur_owner=$(whoami)
    if [ "$cur_owner" != "root" ];then
        return
    fi
    #计算被'/'符号分割的段数
    local seg_num=$(expr $(echo ${1} | grep -o "/" | wc -l) + "1")
    local path=""
    #根据段数遍历所有路径
    for((i=1;i<=$seg_num;i++))
    do
        local split=$(echo ${1} | cut -d "/" -f$i)
        if [ "$split" = "" ];then
            continue
        fi
        local path=${path}"/"${split}
        if [ -d "${path}" ]; then
            local path_owner=$(stat -c %U "${path}")
            if [ "$path_owner" != "root" ]; then
                print "ERROR" "Install failed, install path or its parents path owner [$path_owner] is inconsistent with current user [$cur_owner]."
                exit 1
            fi
        fi
    done
}

function check_path() {
    if [ ! -d "${install_dir}" ]; then
        mkdir -p ${install_dir}
        if [ ! -d "${install_dir}" ]; then
            print "ERROR" "Install failed, [ERROR] create ${install_dir} failed"
            exit 1
        fi
    fi
}

function install_python_whl() {
    cd ${install_dir}
    py_version=$(python3 -c 'import sys; print(sys.version_info[0], ".", sys.version_info[1])' | tr -d ' ')
    py_major_version=${py_version%%.*}
    py_minor_version=${py_version##*.}
    if [[ "$py_major_version" == "3" ]] && { [[ "$py_minor_version" == "10" ]] || [[ "$py_minor_version" == "11" ]]; }; then
        python_interpreter="python3.$py_minor_version"
        print "INFO" "Current Python Interpreter: ${python_interpreter}"
    else
        print "ERROR" "MindIE-LLM python api install failed, please install Python3.10 or Python3.11 first"
        exit 1
    fi
    mindie_llm_wheel_path=$(find $install_dir/bin/ -name mindie_llm*.whl)
    model_wrapper_wheel_path=$(find $install_dir/bin/ -name model_wrapper*.whl)
    llm_manager_python_api_demo_wheel_path=$(find $install_dir/bin/ -name llm_manager_python_api_demo*.whl)
    mie_ops_wheel_path=$(find $install_dir/bin/ -name mie_ops*.whl)

    print "INFO" "Ready to start install mindie_llm at ${mindie_llm_wheel_path}"
    chmod 640 ${log_file}
    $python_interpreter -m pip install ${mindie_llm_wheel_path} --log-file ${log_file} --force-reinstall || \
    { print "ERROR" "Failed to install mindie_llm wheel"; exit 1; }
    if [ -n "$llm_manager_python_api_demo_wheel_path" ]; then
        print "INFO" "Ready to start install llm_manager_python_api_demo at ${llm_manager_python_api_demo_wheel_path}"
        chmod 640 ${log_file}
        $python_interpreter -m pip install ${llm_manager_python_api_demo_wheel_path} --log-file ${log_file} --force-reinstall || \
        { print "ERROR" "Failed to install llm_manager_python_api_demo wheel"; exit 1; }
    else
        print "WARNING" "llm_manager_python_api_demo wheel not found, skipping installation"
    fi
    if [ -n "$mie_ops_wheel_path" ]; then
        chmod 640 ${log_file}
        $python_interpreter -m pip install ${mie_ops_wheel_path} --log-file ${log_file} --force-reinstall || \
        { print "ERROR" "Failed to install mie_ops wheel"; exit 1; }
    else
        print "WARNING" "mie_ops wheel not found, skipping installation"
    fi
}

function install_to_path() {
    install_dir=${default_install_path}/${VERSION}
    if [ -n "${install_dir}" ] && [ -d "${install_dir}" ]; then
        chmod u+w ${default_install_path}
        chmod -R u+w ${install_dir}
        rm -rf $install_dir
    fi
    check_target_dir_owner ${install_dir}
    check_path
    copy_files
    cd ${install_dir}
    install_python_whl
    if [ -f "${default_install_path}/set_env.sh" ]; then
        chmod u+w ${default_install_path}/set_env.sh
        rm -rf ${default_install_path}/set_env.sh
    fi
    mv ${install_dir}/set_env.sh ${default_install_path}
    cd ${default_install_path}
    ln -snf $VERSION latest
}

function copy_files() {
    mkdir -p $install_dir/lib
    mkdir -p $install_dir/bin
    mkdir -p $install_dir/include
    mkdir -p $install_dir/conf
    mkdir -p $install_dir/server/scripts
    cp -r ${sourcedir}/bin/mindieservice_daemon $install_dir/bin
    cp -r ${sourcedir}/*.whl $install_dir/bin
    cp -r ${sourcedir}/scripts $install_dir
    cp -r ${sourcedir}/include/* $install_dir/include
    cp -r ${sourcedir}/conf $install_dir
    cp -r ${sourcedir}/server/scripts/* $install_dir/server/scripts
    cp -r ${sourcedir}/lib/* ${install_dir}/lib
    print "INFO" "debug install ${sourcedir}, install to ${install_dir}"
    cp ${sourcedir}/install.sh $install_dir
    cp ${sourcedir}/version.info $install_dir
    cp ${sourcedir}/set_env.sh $install_dir
}

function install_process() {
    local arch_pkg=MINDIELLMPKGARCH
    ARCH=$(uname -m)
    if [ "${ARCH}" = "x86_64" ]; then
        echo "it is system of x86_64"
    elif [ "${ARCH}" = "aarch64" ]; then
        echo "it is system of aarch64"
    else
        echo "it is not system of aarch64 or x86_64"
    fi
    if [ -n "${ARCH}" ]; then
        if [ "${arch_pkg}" != "${ARCH}" ]; then
            print "ERROR" "Install failed, pkg arch ${arch_pkg} is not consistent with the current environment architecture ${ARCH}."
            exit 1
        fi
    fi
    if [ -n "${target_dir}" ]; then
        if [[ ! "${target_dir}" = /* ]]; then
            print "ERROR" "Install failed, [ERROR] use absolute path for --install-path argument"
            exit 1
        fi
        install_to_path
    else
        install_to_path
    fi
    rm -rf $install_dir/install.sh
}

function check_owner() {
    local cur_owner=$(whoami)
    default_install_path="/usr/local/Ascend/mindie_llm"

    if [ "${ASCEND_HOME_PATH}" == "" ]; then
        print "ERROR" "Install failed, please source cann set_env.sh first."
        exit 1
    else
        cann_path=${ASCEND_HOME_PATH}
    fi

    if [ ! -d "${cann_path}" ]; then
        print "ERROR" "Install failed, can not find cann in ${cann_path}."
        exit 1
    fi
    cann_owner=$(stat -c %U "${cann_path}")

    if [ "${cann_owner}" != "${cur_owner}" ]; then
        print "ERROR" "Install failed, current owner is not same with CANN."
        exit 1
    fi

    if [[ "${cur_owner}" != "root" && "${install_flag}" == "y" ]]; then
        default_install_path="${HOME}/Ascend/mindie_llm"
    fi

    if [ "${install_path_flag}" == "y" ]; then
        default_install_path="${target_dir}"
    fi

    print "INFO" "Check owner success!"
}

function handle_eula() {
    local eula_file=./eula.txt
    if [ ! -f "${eula_file}" ]; then
        print "ERROR" "EULA file ${eula_file} not found"
        exit 1
    fi
    print "INFO" "show ${eula_file}"
    cat "${eula_file}" 1>&2
    read -r -p "Do you accept the EULA to install MindIE-LLM? [Y/n] " answer
    answer=${answer:-Y}
    case ${answer} in
        Y|y)
            print "INFO" "Accept EULA, start to install"
            ;;
        *)
            print "ERROR" "Reject EULA, quit to install"
            exit 1
            ;;
    esac
}

function main() {
    parse_script_args $*
    if [[ "${install_path_flag}" == "y" || "${install_flag}" == "y" ]]; then
        log_init
        check_owner
        if [ "${quiet_flag}" == "n" ]; then
            handle_eula
        else
            print "INFO" "Quiet install, accept EULA by default"
        fi
        install_process
        chmod_authority
        print "INFO" "Ascend-mindie-llm install success!"
    fi
    umask "$ori_umsk" # Restore the original default permission mask.
}

main $*
