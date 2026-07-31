# Configuring a Firewall

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation. In addition, to prevent privilege escalation caused by bringing the environment variables of the current user into other environments during user switch using `su` commands, you should run the `su - [user]` command to switch the user.

Run the `su - [user]` command to switch the user. In addition, add `ALWAYS\_SET\_PATH=yes` to the `/etc/default/su` configuration file of the server to prevent privilege escalation.
