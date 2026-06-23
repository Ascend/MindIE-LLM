# Ensuring that Only the root User Has the Highest Permissions and Each System Account Has a Unique UID

The account whose UID is `0` has the highest permission in a system. Ensure that there is only one such account. In a Linux system, the `root` user is the only super privileged user, and ensure that only the `root` user has a UID of 0. In addition, the UID of each system account must be unique. You can run the `id` command to query it.
