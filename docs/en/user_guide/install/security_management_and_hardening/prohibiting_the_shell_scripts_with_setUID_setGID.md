# Prohibiting the Shell Scripts with the SetUID or SetGID

Scripts with special permissions may be maliciously used, posing great threats to the system. You are advised not to use scripts with the set user ID (SetUID ) and set group ID (SetGID) unless necessary.

Run the following commands to search for files with SetUID or SetGID in the system and check whether they are necessary. If it is not necessary, remove the `s` bit to cancel the SetUID or SetGID permission of the file, or delete the file.

```bash
find / -perm -2000 -exec ls -l {} \; -exec md5sum {} \;
find / -perm -4000 -exec ls -l {} \; -exec md5sum {} \;
```
