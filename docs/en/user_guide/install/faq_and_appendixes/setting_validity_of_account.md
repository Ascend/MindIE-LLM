# Setting User Account Validity Period

To ensure user security, you need to set the validity period of a user. You can run the **chage** command to set the validity period of a user.

Command:

```bash
chage [-m mindays] [-M maxdays] [-d lastday] [-I inactive] [-E expiredate] [-W warndays] user  # typos:ignore
```

For details about related parameters, see [Table 1](#table1).

**Table 1** Setting the validity period of a user <a id="table1"></a>

|Parameter|Description|
|--|--|
|-d--lastday|Displays the date when the valid period was changed the last time.|
|-E--expiredate|Displays the date when a user account expires. The user account is unavailable when the account validity period has expired.|
|-h--help|Displays the help information about the command.|
|-i--iso8601|Changes the expiration date of a user password and displays the date in the YYYY-MM-DD format.|
|-I--inactive|Specifies the inactive period. After the specified time period has expired, a password will be invalid.|
|-l--list|Lists the current settings. It helps non-privileged users to confirm the time when their passwords or accounts expire.|
|-m--mindays|Specifies the minimum number of days before a password can be changed. The value **0** indicates that a password can be changed at any time.|
|-M--maxdays|Specifies the maximum validity period (days) of a password. The value **-1** indicates that the validity check of a password can be disabled. The value **99999** indicates that the validity period is unlimited.|
|-R--root|Sets the root directory where the command is executed to a specified directory.|
|-W--warndays|Specifies the number of days in advance when users are notified that their passwords are about to expire.|

> [!NOTE]
>
>- The date format is YYYY-MM-DD. For example,`chage -E 2017-12-01 test` indicates that the password of the user `test` will expire on December 1, 2017.
>- `user` must be specified. Replace it with the actual user name. The default user is `root`.
>- The account password must be updated periodically. Otherwise, security risks may occur.

For example, to change the validity period of user `test` to 90 days, run the following command:

```bash
chage -M 90 test
```
